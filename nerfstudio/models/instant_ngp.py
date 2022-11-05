"""
Implementation of Instant NGP.
Adapted from the original implementation to allow configuration of more hyperparams (that were previously hard-coded).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from torch.nn.modules.module import T
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch_efficient_distloss import flatten_eff_distloss
import tinycudann as tcnn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 24
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE
    """Resolution of the grid used for the field."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: float = 0.01
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    randomize_background: bool = True
    """Whether to randomize the background color."""

    num_layers_base: int = 2  # Number of layers of the first MLP (both density and RGB)
    hidden_dim_base: int = 64  # Hidden dimensions of first MLP
    geo_feat_dim: int = 15  # Number of geometric features (output from field) that are input for second MLP (only RGB)
    num_layers_color: int = 3  # Number of layers of the second MLP (only RGB)
    hidden_dim_color: int = 64  # Hidden dimensions of second MLP
    appearance_embedding_dim: int = 32  # ?

    n_hashgrid_levels: int = 16
    log2_hashmap_size: int = 19

    lambda_dist_loss: float = 0
    lambda_sparse_prior: float = 0
    lambda_beta_loss: float = 0

    use_background_network: bool = False
    lambda_background_adjustment_regularization: float = 1

    use_spherical_harmonics: bool = True
    latent_dim_time: int = 0
    n_timesteps: int = 1  # Number of timesteps for time embedding


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = TCNNInstantNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            num_layers=self.config.num_layers_base,
            hidden_dim=self.config.hidden_dim_base,
            geo_feat_dim=self.config.geo_feat_dim,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            n_hashgrid_levels=self.config.n_hashgrid_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            use_spherical_harmonics=self.config.use_spherical_harmonics,
            latent_dim_time=self.config.latent_dim_time,
            n_timesteps=self.config.n_timesteps
        )

        if self.config.use_background_network:
            self.mlp_background = tcnn.NetworkWithInputEncoding(
                n_input_dims=6,
                n_output_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "Frequency",
                            "n_frequencies": 12
                        },
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 6
                        }
                    ]

                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": self.config.hidden_dim_color,
                    "n_hidden_layers": 6,
                },
            )
            self.softplus_bg = torch.nn.Softplus()

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler_train = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
            # camera_frustums=self.camera_frustums
        )

        vol_sampler_aabb_eval = None
        if self.config.contraction_type == ContractionType.AABB and self.config.eval_scene_box_scale is not None:
            aabb_scale = self.config.eval_scene_box_scale
            vol_sampler_aabb_eval = torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        self.sampler_eval = VolumetricSampler(
            scene_aabb=vol_sampler_aabb_eval,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
            camera_frustums=self.camera_frustums
        )
        self.sampler = self.sampler_train

        # renderers
        if self.config.use_background_network:
            background_color = None
        elif self.config.randomize_background:
            background_color = 'random'
        else:
            background_color = colors.BLACK

        self.renderer_rgb_train = RGBRenderer(background_color=background_color)
        self.renderer_rgb_eval = RGBRenderer(background_color=colors.BLACK)
        self.renderer_rgb = self.renderer_rgb_train
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.train_step = 0

    # Override train() and eval() to not render random background noise for evaluation
    def eval(self: T) -> T:
        self.renderer_rgb = self.renderer_rgb_eval
        self.sampler = self.sampler_eval

        return super().eval()

    def train(self: T, mode: bool = True) -> T:
        if mode:
            self.renderer_rgb = self.renderer_rgb_train
            self.sampler = self.sampler_train
        else:
            self.renderer_rgb = self.renderer_rgb_eval
            self.sampler = self.sampler_eval

        return super().train(mode)

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        parameters = list(self.field.get_head_parameters())
        parameters.extend(self.field.get_base_parameters())
        parameters.extend(self.mlp_background.parameters())
        param_groups["fields"] = parameters
        # field_head_params = self.field.get_head_parameters()
        # if self.config.use_background_network:
        #     field_head_params.extend(self.mlp_background.parameters())
        # param_groups["field_head"] = field_head_params
        # param_groups["field_base"] = self.field.get_base_parameters()
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, packed_info, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
            "num_samples_per_ray": packed_info[:, 1]
        }

        if self.config.use_background_network:
            # background network

            idx_last_sample_per_ray = (packed_info[:, 0] + packed_info[:, 1] - 1).long()
            t_fars = ray_samples.frustums.ends[idx_last_sample_per_ray]

            background_adjustments = self.mlp_background(
                torch.concat([ray_bundle.origins + t_fars * ray_bundle.directions,
                              ray_bundle.directions],
                             dim=1))  # [R, 3]

            outputs["background_adjustments"] = background_adjustments

        # TODO: should these just always be returned?
        if self.training:
            outputs["ray_samples"] = ray_samples
            outputs["ray_indices"] = ray_indices
            outputs["weights"] = weights
            outputs["n_rays"] = num_rays

        if self.config.lambda_dist_loss > 0 and self.training:
            # Needed for dist loss
            outputs["ray_samples"] = ray_samples
            outputs["ray_indices"] = ray_indices
            outputs["weights"] = weights

        if self.config.lambda_sparse_prior > 0 and self.training:
            outputs["weights"] = weights

        return outputs

    def get_metrics_dict(self, outputs, batch):
        rgb, rgb_without_bg = self._apply_background_network(outputs, batch)

        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(rgb, image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = dict()
        self.train_step += 1

        rgb, rgb_without_bg = self._apply_background_network(outputs, batch)
        outputs["rgb"] = rgb
        if rgb_without_bg is not None:
            outputs["rgb_without_bg"] = rgb_without_bg

        if self.config.use_background_network and "background_adjustments" in outputs:
            background_adjustment_displacement = (outputs["background_adjustments"] - 0.5).pow(2).mean()
            background_adjustment_displacement = self.config.lambda_background_adjustment_regularization * background_adjustment_displacement

            if background_adjustment_displacement.isnan().any():
                print("WARNING! BACKGRUOND ADJUSTMENT REGULARIZATION IS NAN!")

            loss_dict["background_adjustment_displacement"] = background_adjustment_displacement

        rgb_pred = outputs["rgb"]

        image = batch["image"].to(self.device)
        # We removed the masking here to allow gradients to update the background network in transparent regions
        # mask = outputs["alive_ray_mask"]
        # rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])
        rgb_loss = self.rgb_loss(image, rgb_pred)

        loss_dict["rgb_loss"] = rgb_loss

        # print(f"RGB Loss: {rgb_loss.item():0.4f}")

        if self.config.lambda_beta_loss > 0 and self.training:
            # TODO: Make this scheduling more sophisticated, but in principle seems to work
            lambda_beta_loss = self.config.lambda_beta_loss

            accumulation_per_ray = outputs["accumulation"]  # [R]
            beta_loss = ((0.1 + accumulation_per_ray).log() + (1.1 - accumulation_per_ray).log() + 2.20727).mean()
            loss_dict["beta_loss"] = lambda_beta_loss * beta_loss

        if self.config.lambda_dist_loss > 0 and self.training:
            # distloss
            ray_samples = outputs["ray_samples"]
            ray_indices = outputs["ray_indices"]
            weights = outputs["weights"]

            max_samples = 10000
            indices = sorted(
                random.sample(range(len(weights)), min(max_samples, len(weights))))  # Sorting is important!
            ray_indices_small = ray_indices[indices]
            weights_small = weights[indices]
            ends_rays = ray_samples.frustums.ends[indices]
            starts_rays = ray_samples.frustums.starts[indices]

            # TODO: arbitrary near/far ends of rays
            g = lambda x: ((1 / x) - 1 / 5) / (1 / 15 - 1 / 5)
            ends = g(ends_rays)
            starts = g(starts_rays)
            midpoint_distances = (ends + starts) * 0.5
            intervals = ends - starts
            # Need ray_indices for flatten_eff_distloss
            dist_loss = self.config.lambda_dist_loss * flatten_eff_distloss(weights_small,
                                                                            midpoint_distances.squeeze(0),
                                                                            intervals.squeeze(0),
                                                                            ray_indices_small)
            #
            # n_samples = ray_indices_small.shape[0]
            # different_ray_indices = ray_indices_small.bincount()
            # n_rays = different_ray_indices.shape[0]
            # max_samples_per_ray = different_ray_indices.max()
            # weight_combinations = torch.zeros((n_rays, max_samples_per_ray, max_samples_per_ray), device=self.device)
            # midpoint_distances_combinations = torch.zeros((n_rays, max_samples_per_ray, max_samples_per_ray), device=self.device)
            #
            # arange = torch.arange(n_samples, device=self.device).repeat((n_rays, 1))
            # bincount = torch.concat([torch.zeros(1, dtype=torch.int64, device=self.device), different_ray_indices[:-1]])
            # diff_arange = arange - bincount.unsqueeze(1)
            # index_mask = diff_arange >= 0
            # index_mask_2 = torch.concat([index_mask, torch.zeros((1, index_mask.shape[1]), dtype=torch.bool, device=self.device)])
            # insert_indices = index_mask_2.min(dim=0).indices - 1
            #
            # weight_combinations[insert_indices, ray_indices_small] = weights_small
            # midpoint_distances_combinations[insert_indices, ray_indices_small] = midpoint_distances
            #
            #
            # dist_loss = None
            # for ray_id in ray_indices_small.unique():
            #     ray_mask = ray_indices_small == ray_id
            #     ray_weights = weights_small[ray_mask]
            #     ray_midpoint_distances = midpoint_distances[ray_mask].squeeze(1)
            #     n_samples = ray_weights.shape[0]
            #
            #     weight_combinations = ray_weights.repeat((n_samples, 1))
            #     weight_combinations = weight_combinations * weight_combinations.T
            #
            #     midpoint_distances_combinations = ray_midpoint_distances.repeat((n_samples, 1))
            #     midpoint_distances_combinations = (midpoint_distances_combinations - midpoint_distances_combinations.T).abs()
            #
            #     ray_dist_loss = (weight_combinations * midpoint_distances_combinations).sum()
            #     if dist_loss is None:
            #         dist_loss = ray_dist_loss
            #     else:
            #         dist_loss += ray_dist_loss

            # For GPU memory reasons:
            #  1) We only use the uni-lateral regularization on the weights
            #  2) We don't use individual intervals per sample, but assume they are the same for all
            # dist_loss = self.config.lambda_dist_loss * (1/3) * (intervals[0] * weights.pow(2)).sum()
            # print(f"Dist loss: {dist_loss.item():0.4f}")
            loss_dict["dist_loss"] = dist_loss

        if self.config.lambda_sparse_prior > 0 and self.training:
            weights = outputs["weights"]
            accumulation_per_ray = outputs["accumulation"]
            sparsity_loss = self.config.lambda_sparse_prior * accumulation_per_ray.mean()
            # sparsity_loss = self.config.lambda_sparse_prior * (1 + 2 * weights.pow(2)).log().sum()
            loss_dict["sparsity_loss"] = sparsity_loss

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        rgb, rgb_without_bg = self._apply_background_network(outputs, batch)
        outputs["rgb"] = rgb
        if rgb_without_bg is not None:
            outputs["rgb_without_bg"] = rgb_without_bg

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        alive_ray_mask = colormaps.apply_colormap(outputs["alive_ray_mask"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = self.rgb_loss(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()),
                        "ssim": float(ssim),
                        "lpips": float(lpips),
                        "mse": float(mse)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
        }
        if "rgb_without_bg" in outputs:
            images_dict["img_without_bg"] = outputs["rgb_without_bg"]

        return metrics_dict, images_dict

    def _apply_background_network(self,
                                  outputs: Dict[str, torch.Tensor],
                                  batch: Dict[str, torch.Tensor]):
        if "background_images" in batch and not "rgb_without_bg" in outputs:
            background_images = batch["background_images"]  # [B, H, W, 3] or [H, W, 3] (eval)

            if self.training or "local_indices" in batch:
                local_indices = batch["local_indices"]  # [R, 3] with 3 -> (B, H, W)
                background_pixels = background_images[
                    local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]]  # [R, 3]
            else:
                background_pixels = torch.tensor(background_images).to(self.device)  # [H, W, 3]

            if "background_adjustments" in outputs:
                # background_pixels = self.softplus_bg(background_pixels + outputs["background_adjustments"])
                # TODO: subtract -0.5 from background_pixels to make the effort for the bg network symmetric?
                # background_pixels = torch.sigmoid(background_pixels - 0.5 + 10 * outputs["background_adjustments"] - 5)

                ba = outputs["background_adjustments"].mean(dim=-1)
                alpha = (4 * ba.pow(2) - 4 * ba + 1)  # alpha(ba=0|1) -> 1, alpha(ba=0.5) -> 0
                alpha = alpha.unsqueeze(-1)  # [R, 1]
                background_pixels = ((1 - alpha) * background_pixels + alpha * outputs["background_adjustments"])

            rgb_without_bg = outputs["rgb"]
            rgb = outputs["rgb"] + (1 - outputs["accumulation"]) * background_pixels
            # rgb = outputs["rgb"]
        else:
            rgb = outputs["rgb"]
            if "rgb_without_bg" in outputs:
                rgb_without_bg = outputs["rgb_without_bg"]
            else:
                rgb_without_bg = None

        return rgb, rgb_without_bg
