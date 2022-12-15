# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of hypernerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import init
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.generic_scheduler import GenericScheduler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.hypernerf_field import (
    HyperNeRFField,
    HyperSlicingField,
    SE3WarpingField,
)
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.utils import colors, writer


@dataclass
class HyperNeRFModelConfig(VanillaModelConfig):
    """HyperNeRF Model Config"""

    _target: Type = field(default_factory=lambda: HyperNeRFModel)

    collider_type: Literal["AABBBox", "NearFar"] = "NearFar"
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 0.01, "far_plane": 2.0})

    n_freq_pos: int = 9
    n_freq_dir: int = 5
    n_freq_slice: int = 2
    n_layers: int = 8
    hidden_dim: int = 256

    n_timesteps: int = 1
    warp_code_dim: int = 8
    appearance_code_dim: int = 8
    n_cameras: int = 1
    camera_code_dim: int = 8

    use_se3_warping: bool = True
    n_freq_pos_warping: int = 7
    window_alpha_begin: int = 0  # the number of steps window_alpha is set to 0
    window_alpha_end: int = 80000  # the number of steps when window_alpha reaches its maximum

    use_hyper_slicing: bool = True
    n_freq_pos_slicing: int = 7
    hyper_slice_dim: int = 2
    window_beta_begin: int = 1000  # the number of steps window_beta is set to 0
    window_beta_end: int = 10000  # the number of steps when window_beta reaches its maximum


class HyperNeRFModel(NeRFModel):
    """HyperNeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: HyperNeRFModelConfig

    def __init__(
        self,
        config: HyperNeRFModelConfig,
        **kwargs,
    ) -> None:

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""

        base_extra_dim = 0
        head_extra_dim = 0
        self.warp_field = None
        self.slice_field = None
        self.sched_alpha = None
        self.sched_beta = None

        # warp embeddings
        self.warp_embeddings = None
        if self.config.warp_code_dim > 0:
            self.warp_embeddings = nn.Embedding(self.config.n_timesteps, self.config.warp_code_dim)
            init.uniform_(self.warp_embeddings.weight, a=-0.05, b=0.05)

        # appearance embeddings
        self.appearance_embeddings = None
        if self.config.appearance_code_dim > 0:
            self.appearance_embeddings = nn.Embedding(self.config.n_timesteps, self.config.appearance_code_dim)
            init.uniform_(self.appearance_embeddings.weight, a=-0.05, b=0.05)
            head_extra_dim += self.config.appearance_code_dim

        # camera embeddings to model the difference of exposure, color, etc. between cameras
        self.camera_embeddings = None
        if self.config.camera_code_dim > 0:
            self.camera_embeddings = nn.Embedding(self.config.n_cameras, self.config.camera_code_dim)
            init.uniform_(self.camera_embeddings.weight, a=-0.05, b=0.05)
            head_extra_dim += self.config.camera_code_dim

        # fields
        if not self.config.use_hyper_slicing and not self.config.use_se3_warping:
            base_extra_dim = self.config.warp_code_dim

        if self.config.use_se3_warping:
            assert self.warp_embeddings is not None, "SE3WarpingField requires warp_code_dim > 0."
            self.warp_field = SE3WarpingField(
                n_freq_pos=self.config.n_freq_pos_warping,
                warp_code_dim=self.config.warp_code_dim,
                mlp_num_layers=6,
                mlp_layer_width=128,
            )
            if self.config.window_alpha_end >= 1:
                assert self.config.window_alpha_end > self.config.window_alpha_begin
                self.sched_alpha = GenericScheduler(
                    init_value=0,
                    final_value=self.config.n_freq_pos_warping,
                    begin_step=self.config.window_alpha_begin,
                    end_step=self.config.window_alpha_end,
                )

        if self.config.use_hyper_slicing:
            assert self.warp_embeddings is not None, "HyperSlicingField requires warp_code_dim > 0."
            self.slice_field = HyperSlicingField(
                n_freq_pos=self.config.n_freq_pos_slicing,
                out_dim=self.config.hyper_slice_dim,
                warp_code_dim=self.config.warp_code_dim,
                mlp_num_layers=6,
                mlp_layer_width=64,
            )
            if self.config.window_beta_end >= 1:
                assert self.config.window_beta_end > self.config.window_beta_begin
                self.sched_beta = GenericScheduler(
                    init_value=0,
                    final_value=self.config.n_freq_slice,
                    begin_step=self.config.window_beta_begin,
                    end_step=self.config.window_beta_end,
                )

        self.field_coarse = HyperNeRFField(
            n_freq_pos=self.config.n_freq_pos,
            n_freq_dir=self.config.n_freq_dir,
            use_hyper_slicing=self.config.use_hyper_slicing,
            n_freq_slice=self.config.n_freq_slice,
            hyper_slice_dim=self.config.hyper_slice_dim,
            base_extra_dim=base_extra_dim,
            head_extra_dim=head_extra_dim,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
        )

        self.field_fine = HyperNeRFField(
            n_freq_pos=self.config.n_freq_pos,
            n_freq_dir=self.config.n_freq_dir,
            use_hyper_slicing=self.config.use_hyper_slicing,
            n_freq_slice=self.config.n_freq_slice,
            hyper_slice_dim=self.config.hyper_slice_dim,
            base_extra_dim=base_extra_dim,
            head_extra_dim=head_extra_dim,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_uniform = VolumetricSampler(scene_aabb=scene_aabb, )
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # background
        if self.config.use_background_network:
            background_color = None
        elif self.config.randomize_background:
            background_color = "random"
        else:
            background_color = colors.WHITE

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

        # colliders
        if self.config.enable_collider:
            if self.config.collider_type == "AABBBox":
                self.collider = AABBBoxCollider(scene_box=self.scene_box)
            elif self.config.collider_type == "NearFar":
                assert self.config.collider_params is not None
                self.collider = NearFarCollider(
                    near_plane=self.config.collider_params["near_plane"],
                    far_plane=self.config.collider_params["far_plane"],
                )
            else:
                raise NotImplementedError(f"Unkown collider_type: {self.config.collider_type}")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_window_param(sched: GenericScheduler, name: str, step: int):
            sched.update(step)
            writer.put_scalar(name=f"window_param/{name}", scalar=sched.get_value(), step=step)

        callbacks = []

        if self.sched_alpha is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_alpha, "alpha"],
                )
            )

        if self.sched_beta is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_beta, "beta"],
                )
            )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())

        if self.warp_embeddings is not None:
            param_groups["embeddings"] = list(self.warp_embeddings.parameters())

        if self.appearance_embeddings is not None:
            param_groups["embeddings"] = list(self.appearance_embeddings.parameters())

        if self.camera_embeddings is not None:
            param_groups["embeddings"] = list(self.camera_embeddings.parameters())

        if self.warp_field is not None:
            param_groups["fields"] += list(self.warp_field.parameters())

        if self.slice_field is not None:
            param_groups["fields"] += list(self.slice_field.parameters())

        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # window parameters
        if self.sched_alpha is not None:
            window_alpha = self.sched_alpha.get_value() if self.sched_alpha is not None else None
        else:
            window_alpha = None

        if self.sched_beta is not None:
            window_beta = self.sched_beta.get_value() if self.sched_beta is not None else None
        else:
            window_beta = None

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times)
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(
            ray_samples_uniform,
            warp_embeddings=self.warp_embeddings,
            appearance_embeddings=self.appearance_embeddings,
            camera_embeddings=self.camera_embeddings,
            warp_field=self.warp_field,
            slice_field=self.slice_field,
            window_alpha=window_alpha,
            window_beta=window_beta,
        )
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(
            ray_samples_pdf,
            warp_embeddings=self.warp_embeddings,
            appearance_embeddings=self.appearance_embeddings,
            camera_embeddings=self.camera_embeddings,
            warp_field=self.warp_field,
            slice_field=self.slice_field,
            window_alpha=window_alpha,
            window_beta=window_beta,
        )
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }

        if self.training:
            outputs["ray_samples_fine"] = ray_samples_pdf
            outputs["weights_fine"] = weights_fine

        return outputs
