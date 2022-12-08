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
from typing import Any, Dict, List, Tuple, Type

import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.generic_scheduler import GenericScheduler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.hypernerf_field import HyperNeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.utils import colors, writer


@dataclass
class HyperNeRFModelConfig(VanillaModelConfig):
    """HyperNeRF Model Config"""

    _target: Type = field(default_factory=lambda: NeRFModel)

    n_freq_pos: int = 11
    n_freq_dir: int = 5
    n_layers: int = 8
    hidden_dim: int = 256

    n_timesteps: int = 1
    time_embed_dim: int = 8

    use_deformation_field: bool = True
    n_freq_warp: int = 8
    n_freq_slice: int = 6
    window_alpha_begin: int = 0  # the number of steps window_alpha is set to 0
    window_alpha_end: int = 80000  # the number of steps when window_alpha reaches its maximum
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

        if self.config.window_alpha_end >= 1:
            self.sched_alpha = GenericScheduler(
                init_value=0,
                final_value=self.config.n_freq_warp,
                begin_step=self.config.window_alpha_begin,
                end_step=self.config.window_alpha_end,
            )
        else:
            self.sched_alpha = None

        if self.config.window_beta_end >= 1:
            self.sched_beta = GenericScheduler(
                init_value=0,
                final_value=self.config.n_freq_slice,
                begin_step=self.config.window_beta_begin,
                end_step=self.config.window_beta_end,
            )
        else:
            self.sched_beta = None

        # fields
        self.field_coarse = HyperNeRFField(
            n_freq_pos=self.config.n_freq_pos,
            n_freq_dir=self.config.n_freq_dir,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
            n_timesteps=self.config.n_timesteps,
            time_embed_dim=self.config.time_embed_dim,
            use_deformation_field=self.config.use_deformation_field,
            n_freq_warp=self.config.n_freq_warp,
            alpah_sched=self.sched_alpha,
        )

        self.field_fine = HyperNeRFField(
            n_freq_pos=self.config.n_freq_pos,
            n_freq_dir=self.config.n_freq_dir,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
            n_timesteps=self.config.n_timesteps,
            time_embed_dim=self.config.time_embed_dim,
            use_deformation_field=self.config.use_deformation_field,
            n_freq_warp=self.config.n_freq_warp,
            alpah_sched=self.sched_alpha,
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
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

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
