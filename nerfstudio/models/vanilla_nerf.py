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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from elias.config import implicit
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class VanillaModelConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: NeRFModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""

    n_layers: int = 8
    hidden_dim: int = 256

    n_timesteps: int = 1
    latent_dim_time: int = 0


class NeRFModel(Model):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: VanillaModelConfig

    def __init__(
            self,
            config: VanillaModelConfig,
            **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # self.field_coarse = TCNNNeRFField(
        #     position_encoding=position_encoding,
        #     direction_encoding=direction_encoding
        # )
        #
        # self.field_fine = TCNNNeRFField(
        #     position_encoding=position_encoding,
        #     direction_encoding=direction_encoding,
        # )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
            n_timesteps=self.config.n_timesteps,
            latent_dim_time=self.config.latent_dim_time,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.n_layers,
            base_mlp_layer_width=self.config.hidden_dim,
            n_timesteps=self.config.n_timesteps,
            latent_dim_time=self.config.latent_dim_time,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_uniform = VolumetricSampler(scene_aabb=scene_aabb, )
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # background
        if self.config.use_backgrounds:
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
        # TODO: for camera-ready: Add normalize=True
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super(NeRFModel, self).get_param_groups()

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"].extend(self.field_coarse.parameters())
        param_groups["fields"].extend(self.field_fine.parameters())

        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = self.temporal_distortion(ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times)
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
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
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
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

        if self.config.use_background_network:
            # background network
            t_fars = ray_samples_pdf.frustums.ends[:, -1]
            self.apply_background_adjustment(ray_bundle, t_fars, outputs)

        if self.training:
            outputs["ray_samples_fine"] = ray_samples_pdf
            outputs["weights_fine"] = weights_fine

        return outputs

    def get_metrics_dict(self, outputs, batch):
        rgb = self._apply_background_network(batch, outputs)

        # TODO: mask PSNR as well?
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(rgb, image)

        mask = self.get_mask_per_ray(batch)
        if mask is not None:
            metrics_dict["psnr_masked"] = self.psnr(rgb[mask], image[mask])

        floaters = self.get_floaters_metric(batch, outputs["accumulation_fine"])
        if floaters is not None:
            metrics_dict["floaters"] = floaters

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        self._apply_background_network(batch, outputs, overwrite_outputs=True)

        background_adjustment_loss = self.get_background_adjustment_loss(outputs)
        if background_adjustment_loss is not None:
            loss_dict["background_adjustment_displacement"] = background_adjustment_loss

        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.get_masked_rgb_loss(batch, outputs["rgb_fine"])
        mask_loss = self.get_mask_loss(batch, outputs["accumulation_fine"])
        beta_loss = self.get_beta_loss(outputs["accumulation_fine"])

        loss_dict["rgb_loss_coarse"] = rgb_loss_coarse
        loss_dict["rgb_loss_fine"] = rgb_loss_fine
        if mask_loss is not None:
            loss_dict["mask_loss"] = mask_loss
        if beta_loss is not None:
            loss_dict["beta_loss"] = beta_loss

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        self._apply_background_network(batch, outputs, overwrite_outputs=True)

        image = batch["image"].to(self.device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        near_plane = None if self.config.collider_params is None else self.config.collider_params["near_plane"]
        far_plane = None if self.config.collider_params is None else self.config.collider_params["far_plane"]

        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=near_plane,
            far_plane=far_plane,
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=near_plane,
            far_plane=far_plane,
        )

        image_masked, rgb_masked, floaters = self.apply_mask(batch, rgb_fine, acc_fine)

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        mse = self.rgb_loss(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
            "mse": float(mse),
        }

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth
        }

        if image_masked is not None:
            mask = torch.from_numpy(batch["mask"]).squeeze(2)

            combined_rgb_masked = torch.cat([image_masked, rgb_masked], dim=1)

            image_masked = torch.moveaxis(image_masked, -1, 0)[None, ...]
            rgb_masked = torch.moveaxis(rgb_masked, -1, 0)[None, ...]

            psnr_masked = self.psnr(image_masked, rgb_masked)
            # psnr_masked = self.psnr(image_masked[..., mask], rgb_masked[..., mask])
            ssim_masked = self.ssim(image_masked, rgb_masked)
            lpips_masked = self.lpips(image_masked, rgb_masked)
            mse_masked = self.rgb_loss(image_masked, rgb_masked)
            # mse_masked = self.rgb_loss(image_masked[..., mask], rgb_masked[..., mask])

            metrics_dict["psnr_masked"] = float(psnr_masked)
            metrics_dict["ssim_masked"] = float(ssim_masked)
            metrics_dict["lpips_masked"] = float(lpips_masked)
            metrics_dict["mse_masked"] = float(mse_masked)
            metrics_dict["floaters"] = float(floaters)

            images_dict["img_masked"] = combined_rgb_masked

        if "rgb_fine_without_bg" in outputs:
            images_dict["img_without_bg"] = outputs["rgb_fine_without_bg"]

        return metrics_dict, images_dict

    def _apply_background_network(
            self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], overwrite_outputs: bool = False
    ) -> torch.Tensor:

        if self.config.use_backgrounds:

            background_adjustments = outputs["background_adjustments"] if "background_adjustments" in outputs else None

            rgb = self.apply_background_network(
                batch, outputs["rgb_fine"], outputs["accumulation_fine"], background_adjustments
            )

            if overwrite_outputs:
                outputs["rgb_fine_without_bg"] = outputs["rgb_fine"]
                outputs["rgb_fine"] = rgb

        else:
            rgb = outputs["rgb_fine"]

        return rgb
