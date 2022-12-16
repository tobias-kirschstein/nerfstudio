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
Implementation of mip-NeRF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import Parameter, init
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
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc, writer


@dataclass
class MipHyperNeRFModelConfig(ModelConfig):
    """HyperNeRF Model Config"""

    _target: Type = field(default_factory=lambda: MipHyperNerfModel)

    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0})

    collider_type: Literal["AABBBox", "NearFar"] = "NearFar"
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 0.01, "far_plane": 2.0})

    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    use_background_network: bool = False

    use_integrated_encoding: bool = True
    n_freq_pos: int = 17
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
    n_freq_pos_warping: int = 15
    window_alpha_begin: int = 0  # the number of steps window_alpha is set to 0
    window_alpha_end: int = 80000  # the number of steps when window_alpha reaches its maximum

    use_hyper_slicing: bool = True
    n_freq_pos_slicing: int = 15
    hyper_slice_dim: int = 2
    window_beta_begin: int = 1000  # the number of steps window_beta is set to 0
    window_beta_end: int = 10000  # the number of steps when window_beta reaches its maximum


class MipHyperNerfModel(Model):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    config: MipHyperNeRFModelConfig

    def __init__(
        self,
        config: MipHyperNeRFModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        self.warp_field = None
        self.slice_field = None
        self.sched_alpha = None
        self.sched_beta = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        # super().populate_modules()

        # self.field = NeRFField(
        #     position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
        # )

        base_extra_dim = 0
        head_extra_dim = 0

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

        self.field = HyperNeRFField(
            n_freq_pos=self.config.n_freq_pos,
            n_freq_dir=self.config.n_freq_dir,
            use_hyper_slicing=self.config.use_hyper_slicing,
            n_freq_slice=self.config.n_freq_slice,
            hyper_slice_dim=self.config.hyper_slice_dim,
            base_extra_dim=base_extra_dim,
            head_extra_dim=head_extra_dim,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
            use_integrated_encoding=self.config.use_integrated_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # background
        if self.config.use_background_network:
            background_color = None
        else:
            if self.config.background_color == "black":
                background_color = colors.BLACK
            elif self.config.background_color == "white":
                background_color = colors.WHITE
            else:
                background_color = self.config.background_color

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
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())

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

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field is None:
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

        # First pass:
        field_outputs_coarse = self.field.forward(
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

        # Second pass:
        field_outputs_fine = self.field.forward(
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
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=-1, max=1)
        rgb_fine = torch.clip(rgb_fine, min=-1, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
