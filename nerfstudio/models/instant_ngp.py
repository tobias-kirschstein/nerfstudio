"""
Implementation of Instant NGP.
Adapted from the original implementation to allow configuration of more hyperparams (that were previously hard-coded).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import sqrt, ceil
from typing import Dict, List, Optional, Tuple, Type, Union, Literal

import nerfacc
import tinycudann as tcnn
import torch
from elias.config import implicit
from nerfacc import ContractionType
from nerfstudio.field_components.temporal_distortions import SE3Distortion, ViewDirectionWarpType
from nerfstudio.fields.hypernerf_field import HyperSlicingField
from torch import nn
from torch.nn import Parameter, init
from torch.nn.modules.module import T
from torch_efficient_distloss import flatten_eff_distloss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.generic_scheduler import GenericScheduler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, DeformationRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, writer


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
    appearance_embedding_dim: int = 32  # Dimension for every appearance embedding
    use_camera_embedding: bool = False  # Whether a camera-specific code (shared across timesteps) should be learned for RGB network
    camera_embedding_dim: int = 8

    num_layers_base: int = 2  # Number of layers of the first MLP (both density and RGB)
    hidden_dim_base: int = 64  # Hidden dimensions of first MLP
    geo_feat_dim: int = 15  # Number of geometric features (output from field) that are input for second MLP (only RGB)
    num_layers_color: int = 3  # Number of layers of the second MLP (only RGB)
    hidden_dim_color: int = 64  # Hidden dimensions of second MLP

    n_hashgrid_levels: int = 16
    log2_hashmap_size: int = 19
    per_level_hashgrid_scale: float = 1.4472692012786865
    hashgrid_base_resolution: int = 16
    hashgrid_n_features_per_level: int = 2

    lambda_dist_loss: float = 0
    lambda_sparse_prior: float = 0
    lambda_l1_field_regularization: float = 0

    use_spherical_harmonics: bool = True
    disable_view_dependency: bool = False
    latent_dim_time: int = 0
    n_timesteps: int = 1  # Number of timesteps for time embedding
    use_4d_hashing: bool = False
    max_ray_samples_chunk_size: int = -1

    use_deformation_field: bool = False
    n_layers_deformation_field: int = 6
    hidden_dim_deformation_field: int = 128
    use_deformation_hash_encoding_ensemble: bool = False  # Whether to use an ensemble of hash encodings instead of positional encoding
    n_freq_pos_warping: int = 7
    n_freq_pos_ambient: int = 7
    window_deform_begin: int = 0  # the number of steps window_deform is set to 0
    window_deform_end: int = 80000  # the number of steps when window_deform reaches its maximum
    window_ambient_begin: int = 0  # the number of steps window_ambient is set to 0
    window_ambient_end: int = 0  # The number of steps when window_ambient reaches its maximum
    n_ambient_dimensions: int = 0  # How many ambient dimensions should be used
    fix_canonical_space: bool = False  # If True, only canonical space ray can optimize the reconstruction and all other timesteps can only affect the deformation field
    timestep_canonical: Optional[
        int] = 0  # Rays in the canonical timestep won't be deformed, if a deformation field is used
    lambda_deformation_loss: float = 0
    use_time_conditioning_for_base_mlp: bool = False
    use_time_conditioning_for_rgb_mlp: bool = False
    use_deformation_skip_connection: bool = False
    use_smoothstep_hashgrid_interpolation: bool = False
    view_direction_warping: ViewDirectionWarpType = None

    no_hash_encoding: bool = False
    n_frequencies: int = 12

    early_stop_eps: float = 1e-4
    alpha_thre: float = 1e-2
    density_threshold: Optional[float] = None  # if set, densities below the value will be ignored during inference
    view_frustum_culling: Optional[
        int] = None  # Filters out points that are seen by less than the specified number of cameras
    use_view_frustum_culling_for_train: bool = False  # Whether to also filter points during training (slow)


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

            num_images=self.num_train_data,
            num_layers=self.config.num_layers_base,
            hidden_dim=self.config.hidden_dim_base,
            geo_feat_dim=self.config.geo_feat_dim,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,

            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_camera_embedding=self.config.use_camera_embedding,
            camera_embedding_dim=self.config.camera_embedding_dim,

            n_hashgrid_levels=self.config.n_hashgrid_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            per_level_hashgrid_scale=self.config.per_level_hashgrid_scale,
            hashgrid_base_resolution=self.config.hashgrid_base_resolution,
            hashgrid_n_features_per_level=self.config.hashgrid_n_features_per_level,

            use_spherical_harmonics=self.config.use_spherical_harmonics,
            latent_dim_time=self.config.latent_dim_time,
            n_timesteps=self.config.n_timesteps,
            max_ray_samples_chunk_size=self.config.max_ray_samples_chunk_size,

            fix_canonical_space=self.config.fix_canonical_space,
            timestep_canonical=self.config.timestep_canonical,
            use_time_conditioning_for_base_mlp=self.config.use_time_conditioning_for_base_mlp,
            use_time_conditioning_for_rgb_mlp=self.config.use_time_conditioning_for_rgb_mlp,
            use_deformation_skip_connection=self.config.use_deformation_skip_connection,
            use_smoothstep_hashgrid_interpolation=self.config.use_smoothstep_hashgrid_interpolation,

            no_hash_encoding=self.config.no_hash_encoding,
            n_frequencies=self.config.n_frequencies,
            density_threshold=self.config.density_threshold,
            use_4d_hashing=self.config.use_4d_hashing,
            n_ambient_dimensions=self.config.n_ambient_dimensions,
            n_freq_pos_ambient=self.config.n_freq_pos_ambient,
            disable_view_dependency=self.config.disable_view_dependency,

            density_fn_ray_samples_transform=self.warp_ray_samples
        )

        if self.config.use_deformation_field:
            self.temporal_distortion = SE3Distortion(
                self.scene_box.aabb,
                contraction_type=self.config.contraction_type,
                n_freq_pos=self.config.n_freq_pos_warping,
                warp_code_dim=self.config.latent_dim_time,
                mlp_num_layers=self.config.n_layers_deformation_field,
                mlp_layer_width=self.config.hidden_dim_deformation_field,
                view_direction_warping=self.config.view_direction_warping,
                use_hash_encoding_ensemble=self.config.use_deformation_hash_encoding_ensemble)

            self.time_embedding = nn.Embedding(self.config.n_timesteps, self.config.latent_dim_time)
            init.normal_(self.time_embedding.weight, mean=0., std=0.01 / sqrt(self.config.latent_dim_time))

            if self.config.n_ambient_dimensions > 0:
                self.hyper_slicing_network = HyperSlicingField(self.config.n_freq_pos_ambient,
                                                               out_dim=self.config.n_ambient_dimensions,
                                                               warp_code_dim=self.config.latent_dim_time,
                                                               mlp_num_layers=self.config.n_layers_deformation_field,
                                                               mlp_layer_width=self.config.hidden_dim_deformation_field)
        else:
            self.temporal_distortion = None
            self.time_embedding = None

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
            camera_frustums=self.camera_frustums if self.config.use_view_frustum_culling_for_train else None,
            view_frustum_culling=self.config.view_frustum_culling if self.config.use_view_frustum_culling_for_train else None
        )

        vol_sampler_aabb_eval = vol_sampler_aabb
        if self.config.contraction_type == ContractionType.AABB and self.config.eval_scene_box_scale is not None:
            aabb_scale = self.config.eval_scene_box_scale
            vol_sampler_aabb_eval = torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        self.sampler_eval = VolumetricSampler(
            scene_aabb=vol_sampler_aabb_eval,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
            camera_frustums=self.camera_frustums,
            view_frustum_culling=self.config.view_frustum_culling
        )
        self.sampler = self.sampler_train

        if self.config.window_deform_end >= 1:
            self.sched_window_deform = GenericScheduler(
                init_value=0,
                final_value=self.config.n_freq_pos_warping,
                begin_step=self.config.window_deform_begin,
                end_step=self.config.window_deform_end,
            )
        else:
            self.sched_window_deform = None

        if self.config.window_ambient_begin > 0 or self.config.window_ambient_end > 0:
            self.sched_window_ambient = GenericScheduler(
                init_value=0,
                final_value=self.config.n_freq_pos_warping,
                begin_step=self.config.window_ambient_begin,
                end_step=self.config.window_ambient_end,
            )
        else:
            self.sched_window_ambient = None

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
        self.renderer_rgb_train = RGBRenderer(background_color=background_color)
        self.renderer_rgb_eval = RGBRenderer(background_color=background_color)
        self.renderer_rgb = self.renderer_rgb_train
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_deformation = DeformationRenderer()

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
        callbacks = []

        # Occupancy grid
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
                # occ_eval_fn=lambda x: torch.ones_like(x)[..., 0],
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            )
        )

        # Window scheduling for deformation field
        def update_window_param(sched: GenericScheduler, name: str, step: int):
            sched.update(step)
            writer.put_scalar(name=f"window_param/{name}", scalar=sched.get_value(), step=step)

        if self.sched_window_deform is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_deform, "sched_window_deform"],
                )
            )

        if self.sched_window_ambient is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_ambient, "sched_window_ambient"],
                )
            )

        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super(NGPModel, self).get_param_groups()

        field_param_groups = self.field.get_param_groups()
        for key, params in field_param_groups.items():
            if key in param_groups:
                param_groups[key].extend(params)
            else:
                param_groups[key] = params

        # if self.field is None:
        #     raise ValueError("populate_fields() must be called before get_param_groups")
        #
        # param_groups["fields"].extend(self.field.mlp_base.parameters())
        # param_groups["fields"].extend(self.field.mlp_head.parameters())
        #
        # if self.config.use_camera_embedding:
        #     param_groups["fields"].extend(self.field.camera_embedding.parameters())

        if self.temporal_distortion is not None:
            param_groups["deformation_field"] = []
            param_groups["deformation_field"].extend(self.temporal_distortion.parameters())

        if self.time_embedding is not None:
            if "deformation_field" not in param_groups:
                param_groups["deformation_field"] = []

            param_groups["deformation_field"].extend(self.time_embedding.parameters())

        # # TODO: This is from old way of time conditioning
        # if self.field.deformation_network is not None:
        #     param_groups["deformation_field"].extend(self.field.deformation_network.parameters())
        #
        # if self.field.time_embedding is not None:
        #     param_groups["deformation_field"].extend(self.field.time_embedding.parameters())

        # parameters = list(self.field.get_head_parameters())
        # parameters.extend(self.field.get_base_parameters())
        # if self.config.use_background_network:
        #     parameters.extend(self.mlp_background.parameters())
        # if self.config.use_deformation_field:
        #     parameters.extend(self.field.deformation_network.parameters())
        # param_groups["fields"] = parameters

        # field_head_params = self.field.get_head_parameters()
        # if self.config.use_background_network:
        #     field_head_params.extend(self.mlp_background.parameters())
        # param_groups["field_head"] = field_head_params
        # param_groups["field_base"] = self.field.get_base_parameters()
        return param_groups

    def warp_ray_samples(self, ray_samples: RaySamples) -> Tuple[RaySamples, Optional[torch.TensorType]]:
        # window parameters
        if self.sched_window_deform is not None:
            # TODO: Maybe go back to using get_value() which outputs final_value for evaluation
            # window_deform = self.sched_window_deform.get_value() if self.sched_window_deform is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
        else:
            window_deform = None

        time_embeddings = None
        if self.temporal_distortion is not None:

            if ray_samples.timesteps is None:
                # Assume ray_samples come from occupancy grid.
                # We only have one grid to model the whole scene accross time.
                # Hence, we say, only grid cells that are empty for all timesteps should be really empty.
                # Thus, we sample random timesteps for these ray samples
                ray_samples.timesteps = torch.randint(self.config.n_timesteps, (ray_samples.size, 1)).to(
                    ray_samples.frustums.origins.device)

            # Initialize all offsets with 0
            assert ray_samples.frustums.offsets is None, "ray samples have already been warped"
            ray_samples.frustums.offsets = torch.zeros_like(ray_samples.frustums.origins)
            # Need to clone here, as directions was created in a no_grad() block
            ray_samples.frustums.directions = ray_samples.frustums.directions.clone()

            max_chunk_size = ray_samples.size if self.config.max_ray_samples_chunk_size == -1 else self.config.max_ray_samples_chunk_size
            time_embeddings = []

            for i_chunk in range(ceil(ray_samples.size / max_chunk_size)):
                ray_samples_chunk = ray_samples.view(slice(i_chunk * max_chunk_size, (i_chunk + 1) * max_chunk_size))
                timesteps_chunk = ray_samples_chunk.timesteps.squeeze(-1)  # [S]
                time_embeddings_chunk = self.time_embedding(timesteps_chunk)
                time_embeddings.append(time_embeddings_chunk)

                if self.config.timestep_canonical is not None:
                    # Only deform samples that are not from the canonical timestep
                    idx_timesteps_deform = timesteps_chunk != self.config.timestep_canonical

                    if idx_timesteps_deform.any():
                        # Compute offsets
                        time_embeddings_deform = time_embeddings_chunk[idx_timesteps_deform]
                        ray_samples_deform = ray_samples_chunk[idx_timesteps_deform]
                        self.temporal_distortion(ray_samples_deform, warp_code=time_embeddings_deform,
                                                 windows_param=window_deform)

                        # Need to explicitly set offsets because ray_samples_deform contains a copy of the ray samples
                        ray_samples_chunk.frustums.offsets[idx_timesteps_deform] = ray_samples_deform.frustums.offsets
                        ray_samples_chunk.frustums.directions[
                            idx_timesteps_deform] = ray_samples_deform.frustums.directions

                else:
                    # Deform all samples into the latent canonical space
                    self.temporal_distortion(ray_samples_chunk, warp_code=time_embeddings_chunk,
                                             windows_param=window_deform)
                    # ray_samples.frustums.directions[slice(i_chunk * max_chunk_size, (i_chunk + 1) * max_chunk_size)] = ray_samples_chunk.frustums.directions

            time_embeddings = torch.concat(time_embeddings, dim=0)
        return ray_samples, time_embeddings

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
                early_stop_eps=self.config.early_stop_eps,
                alpha_thre=self.config.alpha_thre,
            )

        # window parameters
        if self.sched_window_deform is not None:
            # TODO: Maybe go back to using get_value() which outputs final_value for evaluation
            # window_deform = self.sched_window_deform.get_value() if self.sched_window_deform is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
        else:
            window_deform = None

        ray_samples.ray_indices = ray_indices.unsqueeze(1)  # [S, 1]
        ray_samples, time_codes = self.warp_ray_samples(ray_samples)

        if self.config.n_ambient_dimensions > 0:
            # TODO: maybe move this inside warp_samples?

            window_ambient = None if self.sched_window_ambient is None else self.sched_window_ambient.value

            positions_posed_space = ray_samples.frustums.get_positions(omit_offsets=True, omit_ambient_coordinates=True)
            ambient_coordinates = self.hyper_slicing_network(positions_posed_space,
                                                             warp_code=time_codes,
                                                             window_param=window_ambient)
            ray_samples.frustums.ambient_coordinates = ambient_coordinates

        field_outputs = self.field(ray_samples, window_deform=window_deform, time_codes=time_codes)

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
            "num_samples_per_ray": packed_info[:, 1],
        }

        if ray_samples.frustums.offsets is not None:
            deformation_per_ray = self.renderer_deformation(weights=weights, ray_samples=ray_samples,
                                                            ray_indices=ray_indices, num_rays=num_rays)
            outputs["deformation"] = deformation_per_ray

        if self.config.use_background_network:
            # background network

            idx_last_sample_per_ray = (packed_info[:, 0] + packed_info[:, 1] - 1).long()
            t_fars = ray_samples.frustums.ends[idx_last_sample_per_ray]

            self.apply_background_adjustment(ray_bundle, t_fars, outputs)

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
        rgb = self._apply_background_network(batch, outputs)

        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(rgb, image)

        mask = self.get_mask_per_ray(batch)
        if mask is not None:
            metrics_dict["psnr_masked"] = self.psnr(rgb[mask], image[mask])

        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = dict()
        self.train_step += 1

        self._apply_background_network(batch, outputs, overwrite_outputs=True)

        background_adjustment_loss = self.get_background_adjustment_loss(outputs)
        if background_adjustment_loss is not None:
            loss_dict["background_adjustment_displacement"] = background_adjustment_loss

        rgb_pred = outputs["rgb"]

        # We removed the masking here to allow gradients to update the background network in transparent regions
        # mask = outputs["alive_ray_mask"]
        # rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])

        rgb_loss = self.get_masked_rgb_loss(batch, rgb_pred)
        loss_dict["rgb_loss"] = rgb_loss

        mask_loss = self.get_mask_loss(batch, outputs["accumulation"])
        if mask_loss is not None:
            loss_dict["mask_loss"] = mask_loss

        beta_loss = self.get_beta_loss(outputs["accumulation"])
        if beta_loss is not None:
            loss_dict["beta_loss"] = beta_loss

        if 'deformation' in outputs and self.config.lambda_deformation_loss > 0:
            deformation_loss = self.config.lambda_deformation_loss * (outputs['deformation'] ** 2).mean()
            loss_dict["deformation_loss"] = deformation_loss

        if self.config.lambda_dist_loss > 0 and self.training:
            # distloss
            ray_samples = outputs["ray_samples"]
            ray_indices = outputs["ray_indices"]
            weights = outputs["weights"]

            max_samples = 10000
            indices = sorted(
                random.sample(range(len(weights)), min(max_samples, len(weights)))
            )  # Sorting is important!
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
            dist_loss = self.config.lambda_dist_loss * flatten_eff_distloss(
                weights_small, midpoint_distances.squeeze(0), intervals.squeeze(0), ray_indices_small
            )
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

        # TODO: L1 regularization for hash table (Is inside mlp_base)
        if self.config.lambda_l1_field_regularization > 0:
            loss_dict["l1_field_regularization"] = (
                    self.config.lambda_l1_field_regularization * self.field.mlp_base.params.abs().mean()
            )

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        self._apply_background_network(batch, outputs, overwrite_outputs=True)

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        alive_ray_mask = colormaps.apply_colormap(outputs["alive_ray_mask"])

        image_masked, rgb_masked, floaters = self.apply_mask(batch, rgb, acc)

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)
        error_image = ((rgb - image) ** 2).mean(dim=-1).unsqueeze(-1)
        error_image = colormaps.apply_colormap(error_image, cmap="turbo")

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = self.rgb_loss(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips),
            "mse": float(mse),
        }  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
            "error": error_image
        }

        if "deformation" in outputs:
            deformation_img = colormaps.apply_offset_colormap(outputs["deformation"])
            images_dict["deformation"] = deformation_img

        if image_masked is not None:
            mask = torch.from_numpy(batch["mask"]).squeeze(2)

            combined_rgb_masked = torch.cat([image_masked, rgb_masked], dim=1)

            image_masked = torch.moveaxis(image_masked, -1, 0)[None, ...]
            rgb_masked = torch.moveaxis(rgb_masked, -1, 0)[None, ...]

            psnr_masked = self.psnr(image_masked[..., mask], rgb_masked[..., mask])
            ssim_masked = self.ssim(image_masked, rgb_masked)
            lpips_masked = self.lpips(image_masked, rgb_masked)
            mse_masked = self.rgb_loss(image_masked[..., mask], rgb_masked[..., mask])

            metrics_dict["psnr_masked"] = psnr_masked
            metrics_dict["ssim_masked"] = ssim_masked
            metrics_dict["lpips_masked"] = lpips_masked
            metrics_dict["mse_masked"] = mse_masked
            metrics_dict["floaters"] = float(floaters)

            images_dict["img_masked"] = combined_rgb_masked

        if "rgb_without_bg" in outputs:
            images_dict["img_without_bg"] = outputs["rgb_without_bg"]

        return metrics_dict, images_dict

    def _apply_background_network(self,
                                  batch: Dict[str, torch.Tensor],
                                  outputs: Dict[str, torch.Tensor],
                                  overwrite_outputs: bool = False) -> torch.Tensor:
        if self.config.use_backgrounds:
            background_adjustments = outputs["background_adjustments"] if "background_adjustments" in outputs else None
            rgb = self.apply_background_network(batch,
                                                outputs["rgb"],
                                                outputs["accumulation"],
                                                background_adjustments)

            if overwrite_outputs:
                outputs["rgb_without_bg"] = outputs["rgb"]
                outputs["rgb"] = rgb
        else:
            rgb = outputs["rgb"]

        return rgb
