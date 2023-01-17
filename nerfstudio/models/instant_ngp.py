"""
Implementation of Instant NGP.
Adapted from the original implementation to allow configuration of more hyperparams (that were previously hard-coded).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
import tinycudann as tcnn
import torch
from elias.config import implicit
from nerfacc import ContractionType, contract
from torch import TensorType, nn
from torch.nn import Parameter, init
from torch.nn.modules.module import T, _IncompatibleKeys
from torch_efficient_distloss import flatten_eff_distloss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.generic_scheduler import GenericScheduler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.hash_encoding import HashEnsembleMixingType
from nerfstudio.field_components.occupancy import FilteredOccupancyGrid
from nerfstudio.field_components.temporal_distortions import (
    SE3Distortion,
    ViewDirectionWarpType,
)
from nerfstudio.fields.hypernerf_field import HyperSlicingField
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DeformationRenderer,
    DepthRenderer,
    RGBRenderer,
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
    use_affine_color_transformation: bool = False  # If set, camera embedding won't directly condition RGB MLP but instead is used to decode an affine transformation of the predicted RGB values

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
    lambda_random_dist_loss: float = 0  # Additionally, shoot random rays in scene and enforce dist loss to be small
    dist_loss_max_rays: int = 500  # To avoid massive GPU memory consumption, one can compute the distloss only on a subset of rays per iteration
    lambda_sparse_prior: float = 0
    lambda_l1_field_regularization: float = 0
    lambda_global_sparsity_prior: float = 0  # Force density globally to be as small as possible

    lambda_temporal_tv_loss: float = 0  # enforce total variation loss across temporal codes

    use_spherical_harmonics: bool = True
    spherical_harmonics_degree: int = 4
    disable_view_dependency: bool = False
    latent_dim_time: int = 0
    latent_dim_time_deformation: Optional[
        int] = None  # In case multiple time embeddings are used, can specify a different dimension for deformation field
    n_timesteps: int = 1  # Number of timesteps for time embedding
    use_4d_hashing: bool = False
    use_hash_encoding_ensemble: bool = False  # Whether to use an ensemble of hash encodings for the canonical space
    max_ray_samples_chunk_size: int = -1
    hash_encoding_ensemble_n_levels: int = 16
    hash_encoding_ensemble_features_per_level: int = 2
    hash_encoding_ensemble_n_tables: Optional[int] = None,
    hash_encoding_ensemble_mixing_type: HashEnsembleMixingType = 'blend'
    hash_encoding_ensemble_n_heads: Optional[int] = None  # If None, will use the same as n_tables
    hash_encoding_ensemble_disable_initial: bool = False  # If set and window_hash_tables_end is used, the single hash table in the beginning will be a plain Instant NGP (without multiplying with the respective time code), forcing the network to use the deformation field
    hash_encoding_ensemble_disable_table_chunking: bool = False  # Backward compatibility, disables performance improvement that chunks hashtables together
    hash_encoding_ensemble_use_soft_transition: bool = False  # If disable_initial is used, slow transition ensures that there is no sudden jump in the blend weight for the first hashtable once window_hash_tables_begin is reached
    hash_encoding_ensemble_swap_l_f: bool = False  # Deprecated, if enabled, features will be passed as FxL instead of LxF to the base MLP (L=layers, F=features)

    blend_field_hidden_dim: int = 64
    blend_field_n_freq_enc: int = 0
    blend_field_skip_connections: Optional[Tuple[int]] = None
    window_blend_end: int = 0

    use_deformation_field: bool = False
    n_layers_deformation_field: int = 6
    hidden_dim_deformation_field: int = 128
    use_deformation_hash_encoding_ensemble: bool = False  # Whether to use an ensemble of hash encodings instead of positional encoding for the deformation field
    n_freq_pos_warping: int = 7
    n_freq_pos_ambient: int = 7
    use_hash_se3field: bool = False

    window_deform_begin: int = 0  # the number of steps window_deform is set to 0
    window_deform_end: int = 80000  # the number of steps when window_deform reaches its maximum
    window_ambient_begin: int = 0  # the number of steps window_ambient is set to 0
    window_ambient_end: int = 0  # The number of steps when window_ambient reaches its maximum
    window_canonical_begin: int = 0  # Iteration at which allowing more complexity for canonical space should be started
    window_canonical_end: int = 0  # Iteration at which all complexity for canonical space should be there
    window_canonical_initial: int = 3  # How many levels of the canonical hash encoding should be active initially
    window_hash_tables_begin: int = 0  # Step when more hash tables should be gradually added
    window_hash_tables_end: int = 0  # Step when all hash tables should be present

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
    alpha_thre: float = 1e-2  # Threshold for skipping empty space when sampling points on rays
    occ_thre: float = 1e-2  # Threshold for how large density needs to be such that occupancy grid marks region as occupied
    density_threshold: Optional[float] = None  # if set, densities below the value will be ignored during inference
    view_frustum_culling: Optional[
        int] = None  # Filters out points that are seen by less than the specified number of cameras
    use_view_frustum_culling_for_train: bool = False  # Whether to also filter points during training (slow)
    use_occupancy_grid_filtering: bool = False  # If enabled, filters the occupancy grid to only contain the largest connected component which should remove floaters

    only_render_canonical_space: bool = False  # Special option for evaluation purposes: Disables any existing deformation field
    only_render_hash_table: Optional[
        int] = None  # Special option for evaluation purpose: Only query specified hash table in Hash Ensembles


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
            use_affine_color_transformation=self.config.use_affine_color_transformation,

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

            use_hash_encoding_ensemble=self.config.use_hash_encoding_ensemble,
            hash_encoding_ensemble_n_levels=self.config.hash_encoding_ensemble_n_levels,
            hash_encoding_ensemble_features_per_level=self.config.hash_encoding_ensemble_features_per_level,
            hash_encoding_ensemble_n_tables=self.config.hash_encoding_ensemble_n_tables,
            hash_encoding_ensemble_mixing_type=self.config.hash_encoding_ensemble_mixing_type,
            hash_encoding_ensemble_n_heads=self.config.hash_encoding_ensemble_n_heads,
            hash_encoding_ensemble_disable_initial=self.config.hash_encoding_ensemble_disable_initial,
            hash_encoding_ensemble_disable_table_chunking=self.config.hash_encoding_ensemble_disable_table_chunking,
            hash_encoding_ensemble_use_soft_transition=self.config.hash_encoding_ensemble_use_soft_transition,
            hash_encoding_ensemble_swap_l_f=self.config.hash_encoding_ensemble_swap_l_f,
            only_render_hash_table=self.config.only_render_hash_table,
            blend_field_skip_connections=self.config.blend_field_skip_connections,
            n_freq_pos_warping=self.config.n_freq_pos_warping,

            blend_field_hidden_dim=self.config.blend_field_hidden_dim,
            blend_field_n_freq_enc=self.config.blend_field_n_freq_enc,

            density_fn_ray_samples_transform=self.warp_ray_samples
        )

        self.temporal_distortion = None
        self.time_embedding = None
        if self.config.use_deformation_field or self.config.use_hash_encoding_ensemble:
            self.time_embedding = nn.Embedding(self.config.n_timesteps, self.config.latent_dim_time)
            init.normal_(self.time_embedding.weight, mean=0., std=0.01 / sqrt(self.config.latent_dim_time))

        if self.config.use_deformation_field \
                and self.config.latent_dim_time_deformation is not None \
                and self.config.use_hash_encoding_ensemble:
            # If both deformation field AND canonical hash ensemble are used, can have separate time embeddings
            self.use_separate_deformation_time_embedding = True
            self.time_embedding_deformation = nn.Embedding(self.config.n_timesteps,
                                                           self.config.latent_dim_time_deformation)
            init.normal_(self.time_embedding_deformation.weight, mean=0.,
                         std=0.01 / sqrt(self.config.latent_dim_time_deformation))
            latent_dim_time_deformation = self.config.latent_dim_time_deformation
        else:
            self.use_separate_deformation_time_embedding = False
            self.time_embedding_deformation = None
            latent_dim_time_deformation = self.config.latent_dim_time

        if self.config.use_deformation_field:
            self.temporal_distortion = SE3Distortion(
                self.scene_box.aabb,
                contraction_type=self.config.contraction_type,
                n_freq_pos=self.config.n_freq_pos_warping,
                warp_code_dim=latent_dim_time_deformation,
                mlp_num_layers=self.config.n_layers_deformation_field,
                mlp_layer_width=self.config.hidden_dim_deformation_field,
                view_direction_warping=self.config.view_direction_warping,
                use_hash_se3field=self.config.use_hash_se3field,
                use_hash_encoding_ensemble=self.config.use_deformation_hash_encoding_ensemble,
                hash_encoding_ensemble_n_levels=self.config.hash_encoding_ensemble_n_levels,
                hash_encoding_ensemble_features_per_level=self.config.hash_encoding_ensemble_features_per_level,
                hash_encoding_ensemble_n_tables=self.config.hash_encoding_ensemble_n_tables,
                hash_encoding_ensemble_mixing_type=self.config.hash_encoding_ensemble_mixing_type,
                hash_encoding_ensemble_n_heads=self.config.hash_encoding_ensemble_n_heads,
                only_render_hash_table=self.config.only_render_hash_table,
            )

            if self.config.n_ambient_dimensions > 0:
                self.hyper_slicing_network = HyperSlicingField(self.config.n_freq_pos_ambient,
                                                               out_dim=self.config.n_ambient_dimensions,
                                                               warp_code_dim=latent_dim_time_deformation,
                                                               mlp_num_layers=self.config.n_layers_deformation_field,
                                                               mlp_layer_width=self.config.hidden_dim_deformation_field)

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        if self.config.use_occupancy_grid_filtering:
            self.occupancy_grid_eval = FilteredOccupancyGrid(self.occupancy_grid)
        else:
            self.occupancy_grid_eval = self.occupancy_grid

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler_train = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.density_fn,
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
            occupancy_grid=self.occupancy_grid_eval,
            density_fn=self.density_fn,
            camera_frustums=self.camera_frustums,
            view_frustum_culling=self.config.view_frustum_culling
        )
        self.sampler = self.sampler_train

        if self.config.window_deform_end >= 1:
            self.sched_window_deform = GenericScheduler(
                init_value=0,
                final_value=self.config.hash_encoding_ensemble_n_levels if self.config.use_deformation_hash_encoding_ensemble else self.config.n_freq_pos_warping,
                begin_step=self.config.window_deform_begin,
                end_step=self.config.window_deform_end,
            )
        else:
            self.sched_window_deform = None

        if self.config.landmark_loss_end is not None:
            self.sched_landmark_loss = GenericScheduler(
                init_value=self.config.lambda_landmark_loss,
                final_value=0,
                begin_step=0,
                end_step=self.config.landmark_loss_end,
            )
        else:
            self.sched_landmark_loss = None

        if self.config.window_ambient_begin > 0 or self.config.window_ambient_end > 0:
            self.sched_window_ambient = GenericScheduler(
                init_value=0,
                final_value=self.config.n_freq_pos_warping,
                begin_step=self.config.window_ambient_begin,
                end_step=self.config.window_ambient_end,
            )
        else:
            self.sched_window_ambient = None

        if self.config.use_hash_encoding_ensemble and self.config.window_canonical_end >= 1:
            self.sched_window_canonical = GenericScheduler(
                init_value=self.config.window_canonical_initial,
                final_value=self.config.hash_encoding_ensemble_n_levels,
                begin_step=self.config.window_canonical_begin,
                end_step=self.config.window_canonical_end,
            )
        else:
            self.sched_window_canonical = None

        if self.config.use_hash_encoding_ensemble and self.config.window_blend_end > 0:
            self.sched_window_blend = GenericScheduler(
                init_value=0,
                final_value=self.config.blend_field_n_freq_enc,
                begin_step=0,
                end_step=self.config.window_blend_end,
            )
        else:
            self.sched_window_blend = None

        if self.config.use_hash_encoding_ensemble and self.config.window_hash_tables_end > 0:
            self.sched_window_hash_tables = GenericScheduler(
                init_value=1,
                final_value=self.field.hash_encoding_ensemble.n_hash_encodings,
                begin_step=self.config.window_hash_tables_begin,
                end_step=self.config.window_hash_tables_end,
            )
        else:
            self.sched_window_hash_tables = None

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

        self.register_load_state_dict_post_hook(self.load_state_dict_post_hook)
        self._fixed_view_direction = None

    # Override train() and eval() to not render random background noise for evaluation
    def eval(self: T) -> T:
        self.renderer_rgb = self.renderer_rgb_eval
        self.sampler = self.sampler_eval

        if self.config.use_occupancy_grid_filtering:
            # Whenever we change to eval() mode, update the eval occupancy grid s.t. it only contains the largest
            # connected component
            self.occupancy_grid_eval.update()

        return super().eval()

    def train(self: T, mode: bool = True) -> T:
        if mode:
            self.renderer_rgb = self.renderer_rgb_train
            self.sampler = self.sampler_train
        else:
            self.renderer_rgb = self.renderer_rgb_eval
            self.sampler = self.sampler_eval

            if self.config.use_occupancy_grid_filtering:
                # Whenever we change to eval() mode, update the eval occupancy grid s.t. it only contains the largest
                # connected component
                self.occupancy_grid_eval.update()

        return super().train(mode)

    def density_fn(self, positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
        )

        ray_samples, _ = self.warp_ray_samples(ray_samples)
        if ray_samples.timesteps is not None:
            time_codes = self.time_embedding(ray_samples.timesteps.squeeze(1))
        else:
            time_codes = None

        window_canonical = self.sched_window_canonical.value if self.sched_window_canonical is not None else None
        window_blend = self.sched_window_blend.value if self.sched_window_blend is not None else None
        window_hash_tables = self.sched_window_hash_tables.value if self.sched_window_hash_tables is not None else None
        window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None

        density, _ = self.field.get_density(ray_samples,
                                            window_canonical=window_canonical,
                                            window_blend=window_blend,
                                            window_hash_tables=window_hash_tables,
                                            window_deform=window_deform,
                                            time_codes=time_codes)
        return density

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
                # occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
                occ_eval_fn=lambda x: self.density_fn(x) * self.config.render_step_size,
                occ_thre=self.config.occ_thre,
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

        if self.sched_window_canonical is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_canonical, "sched_window_canonical"],
                )
            )

        if self.sched_window_blend is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_blend, "sched_window_blend"],
                )
            )

        if self.sched_window_hash_tables is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_window_param,
                    args=[self.sched_window_hash_tables, "sched_window_hash_tables"],
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
            param_groups["deformation_field"].extend(self.temporal_distortion.parameters())

        if self.time_embedding is not None:
            if self.use_separate_deformation_time_embedding:
                param_groups["embeddings"].extend(self.time_embedding_deformation.parameters())
                param_groups["embeddings"].extend(self.time_embedding.parameters())
            else:
                param_groups["embeddings"].extend(self.time_embedding.parameters())

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
        if self.config.only_render_canonical_space:
            return ray_samples, None

        # window parameters
        if self.sched_window_deform is not None:
            # TODO: Maybe go back to using get_value() which outputs final_value for evaluation
            # window_deform = self.sched_window_deform.get_value() if self.sched_window_deform is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
        else:
            window_deform = None

        time_embeddings = None
        if self.temporal_distortion is not None or self.config.use_hash_encoding_ensemble:

            if ray_samples.timesteps is None:
                # Assume ray_samples come from occupancy grid.
                # We only have one grid to model the whole scene accross time.
                # Hence, we say, only grid cells that are empty for all timesteps should be really empty.
                # Thus, we sample random timesteps for these ray samples
                ray_samples.timesteps = torch.randint(self.config.n_timesteps, (ray_samples.size, 1)).to(
                    ray_samples.frustums.origins.device)

        if self.temporal_distortion is not None:
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
                if self.use_separate_deformation_time_embedding:
                    time_embeddings_chunk = self.time_embedding_deformation(timesteps_chunk)
                else:
                    time_embeddings_chunk = self.time_embedding(timesteps_chunk)
                time_embeddings.append(time_embeddings_chunk)

                if self.config.timestep_canonical is not None:
                    # Only deform samples that are not from the canonical timestep
                    idx_timesteps_deform = timesteps_chunk != self.config.timestep_canonical

                    if idx_timesteps_deform.any():
                        # Compute offsets
                        time_embeddings_deform = time_embeddings_chunk[idx_timesteps_deform]
                        ray_samples_deform = ray_samples_chunk[idx_timesteps_deform]
                        self.temporal_distortion(ray_samples_deform,
                                                 warp_code=time_embeddings_deform,
                                                 windows_param=window_deform)

                        # Need to explicitly set offsets because ray_samples_deform contains a copy of the ray samples
                        ray_samples_chunk.frustums.offsets[idx_timesteps_deform] = ray_samples_deform.frustums.offsets
                        ray_samples_chunk.frustums.directions[
                            idx_timesteps_deform] = ray_samples_deform.frustums.directions

                else:
                    # Deform all samples into the latent canonical space
                    self.temporal_distortion(ray_samples_chunk,
                                             warp_code=time_embeddings_chunk,
                                             windows_param=window_deform)
                    # ray_samples.frustums.directions[slice(i_chunk * max_chunk_size, (i_chunk + 1) * max_chunk_size)] = ray_samples_chunk.frustums.directions

            time_embeddings = torch.concat(time_embeddings, dim=0)
        elif self.config.use_hash_encoding_ensemble:
            time_embeddings = self.time_embedding(ray_samples.timesteps.squeeze(-1))

        if self.config.n_ambient_dimensions > 0:
            # TODO: maybe move this inside warp_samples?

            window_ambient = None if self.sched_window_ambient is None else self.sched_window_ambient.value

            positions_posed_space = ray_samples.frustums.get_positions(omit_offsets=True, omit_ambient_coordinates=True)
            ambient_coordinates = self.hyper_slicing_network(positions_posed_space,
                                                             warp_code=time_embeddings,
                                                             window_param=window_ambient)
            ray_samples.frustums.ambient_coordinates = ambient_coordinates

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

        ray_samples.ray_indices = ray_indices.unsqueeze(1)  # [S, 1]
        ray_samples, time_codes_deform = self.warp_ray_samples(ray_samples)

        if self.use_separate_deformation_time_embedding and ray_samples.timesteps is not None and self.time_embedding is not None:
            # This potentially uses a different time embedding for the canonical field than the deformation field
            time_codes = self.time_embedding(ray_samples.timesteps.squeeze(1))
        else:
            time_codes = time_codes_deform

        window_canonical = self.sched_window_canonical.value if self.sched_window_canonical is not None else None
        window_blend = self.sched_window_blend.value if self.sched_window_blend is not None else None
        window_hash_tables = self.sched_window_hash_tables.value if self.sched_window_hash_tables is not None else None
        window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None

        field_outputs = self.field(ray_samples,
                                   window_canonical=window_canonical,
                                   window_blend=window_blend,
                                   window_hash_tables=window_hash_tables,
                                   window_deform=window_deform,
                                   time_codes=time_codes,
                                   fixed_view_direction=self._fixed_view_direction)

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

        floaters = self.get_floaters_metric(batch, outputs["accumulation"])
        if floaters is not None:
            metrics_dict["floaters"] = floaters

        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()

        if "landmarks" in batch and self.temporal_distortion is not None:
            with torch.no_grad():
                metrics_dict["landmark_loss"] = self.get_landmark_loss(batch).mean()
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

        alpha_loss = self.get_alpha_loss(batch, outputs["accumulation"])
        if alpha_loss is not None:
            loss_dict["alpha_loss"] = alpha_loss

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

            max_rays = self.config.dist_loss_max_rays
            indices = (ray_indices.unsqueeze(1) == ray_indices.unique()[:max_rays]).any(dim=1)

            ray_indices_small = ray_indices[indices]
            weights_small = weights[indices]
            ends = ray_samples.frustums.ends[indices].squeeze(-1)
            starts = ray_samples.frustums.starts[indices].squeeze(-1)

            midpoint_distances = (ends + starts) * 0.5
            intervals = ends - starts

            # Need ray_indices for flatten_eff_distloss
            dist_loss = self.config.lambda_dist_loss * flatten_eff_distloss(
                weights_small, midpoint_distances, intervals, ray_indices_small
            )

            loss_dict["dist_loss"] = dist_loss

            if self.config.lambda_random_dist_loss > 0:
                typical_origin_distance = ray_samples.frustums.origins.norm(dim=1).mean()
                typical_start = starts.min()  # TODO: could use the mean() of the starting points of rays
                typical_end = ends.max()
                nears = typical_start.repeat(max_rays).unsqueeze(1)
                fars = typical_end.repeat(max_rays).unsqueeze(1)

                random_origins = torch.randn((max_rays, 3), device=weights.device)
                random_origins = random_origins / (
                            random_origins.norm(dim=1).unsqueeze(1) + 1e-8)  # distribute on unit sphere

                directions = -random_origins  # random rays should all point towards 0
                random_directions_offset = torch.randn((max_rays, 3), device=weights.device)
                random_directions_offset = random_directions_offset / (
                            random_directions_offset.norm(dim=1).unsqueeze(1) + 1e-8)  # distribute on unit sphere
                directions = directions + 0.5 * random_directions_offset  # 0.5 -> ray directions are distorted up to a 90° cone
                directions = directions / (directions.norm(dim=1).unsqueeze(1) + 1e-8)
                random_origins = typical_origin_distance * random_origins  # Move points out to roughly match where train cameras are

                random_timesteps = torch.randint(self.config.n_timesteps, (max_rays, 1), dtype=torch.int,
                                                 device=weights.device)

                random_ray_bundle = RayBundle(
                    random_origins,
                    directions,
                    None,
                    camera_indices=None,
                    nears=nears,
                    fars=fars,
                    timesteps=random_timesteps
                )

                with torch.no_grad():
                    random_ray_samples, random_packed_info, random_ray_indices = self.sampler(
                        ray_bundle=random_ray_bundle,
                        near_plane=self.config.near_plane,
                        far_plane=self.config.far_plane,
                        render_step_size=self.config.render_step_size,
                        cone_angle=self.config.cone_angle,
                        early_stop_eps=self.config.early_stop_eps,
                        alpha_thre=self.config.alpha_thre,
                    )

                window_canonical = self.sched_window_canonical.value if self.sched_window_canonical is not None else None
                window_blend = self.sched_window_blend.value if self.sched_window_blend is not None else None
                window_hash_tables = self.sched_window_hash_tables.value if self.sched_window_hash_tables is not None else None
                window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
                if random_ray_samples.timesteps is not None:
                    # Detach time codes as we only want to supervise the actual field
                    time_codes = self.time_embedding(random_ray_samples.timesteps.squeeze(1)).detach()
                else:
                    time_codes = None

                density, _ = self.field.get_density(random_ray_samples,
                                                    window_canonical=window_canonical,
                                                    window_blend=window_blend,
                                                    window_hash_tables=window_hash_tables,
                                                    window_deform=window_deform,
                                                    time_codes=time_codes)

                random_weights = nerfacc.render_weight_from_density(
                    packed_info=random_packed_info,
                    sigmas=density,
                    t_starts=random_ray_samples.frustums.starts,
                    t_ends=random_ray_samples.frustums.ends,
                )

                random_midpoint_distances = (
                                                        random_ray_samples.frustums.starts + random_ray_samples.frustums.ends) * 0.5
                random_midpoint_distances = random_midpoint_distances.squeeze(1)
                random_intervals = random_ray_samples.frustums.ends - random_ray_samples.frustums.starts
                random_intervals = random_intervals.squeeze(1)

                # Need ray_indices for flatten_eff_distloss
                random_dist_loss = self.config.lambda_dist_loss * flatten_eff_distloss(
                    random_weights, random_midpoint_distances, random_intervals, random_ray_indices
                )

                loss_dict["random_dist_loss"] = random_dist_loss

        if self.config.lambda_sparse_prior > 0 and self.training:
            weights = outputs["weights"]
            accumulation_per_ray = outputs["accumulation"]
            sparsity_loss = self.config.lambda_sparse_prior * accumulation_per_ray.mean()
            # sparsity_loss = self.config.lambda_sparse_prior * (1 + 2 * weights.pow(2)).log().sum()
            loss_dict["sparsity_loss"] = sparsity_loss

        if self.config.lambda_global_sparsity_prior > 0:
            n_random_points = 128
            random_points = torch.rand(n_random_points, 3) * 2 - 1  # [-1, 1]
            random_points = random_points.to(rgb_pred)
            ray_samples_random = RaySamples(
                Frustums(
                    origins=random_points,
                    directions=torch.zeros_like(random_points),
                    starts=torch.zeros((*random_points.shape[:-1], 1)).to(random_points),
                    ends=torch.zeros((*random_points.shape[:-1], 1)).to(random_points),
                    pixel_area=None
                ),
                timesteps=torch.randint(self.config.n_timesteps, (n_random_points, 1), dtype=torch.int,
                                        device=random_points.device)
            )
            window_canonical = self.sched_window_canonical.value if self.sched_window_canonical is not None else None
            window_blend = self.sched_window_blend.value if self.sched_window_blend is not None else None
            window_hash_tables = self.sched_window_hash_tables.value if self.sched_window_hash_tables is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None

            if ray_samples_random.timesteps is not None:
                # Detach time codes as we only want to supervise the actual field
                time_codes = self.time_embedding(ray_samples_random.timesteps.squeeze(1)).detach()
            else:
                time_codes = None

            density, _ = self.field.get_density(ray_samples_random,
                                                window_canonical=window_canonical,
                                                window_blend=window_blend,
                                                window_hash_tables=window_hash_tables,
                                                window_deform=window_deform,
                                                time_codes=time_codes)
            assert density.min() >= 0
            global_sparsity_loss = density.mean()
            loss_dict["global_sparsity_loss"] = self.config.lambda_global_sparsity_prior * global_sparsity_loss

        # TODO: L1 regularization for hash table (Is inside mlp_base)
        if self.config.lambda_l1_field_regularization > 0:
            loss_dict["l1_field_regularization"] = (
                    self.config.lambda_l1_field_regularization * self.field.mlp_base.params.abs().mean()
            )

        if "ray_samples" in outputs and self.config.lambda_deformation_l1_prior > 0 and self.train_step < 100000:
            loss_dict["deformation_l1_prior"] = self.config.lambda_deformation_l1_prior * outputs[
                "ray_samples"].frustums.offsets.abs().mean()

        if self.temporal_distortion is not None and self.config.lambda_landmark_loss > 0 and self.train_step < 100000:
            landmark_loss = self.get_landmark_loss(batch)
            loss_dict["landmark_loss"] = (self.sched_landmark_loss.value if
                                          self.sched_landmark_loss is not None
                                          else self.config.lambda_landmark_loss) * \
                                         landmark_loss.mean()

        if self.config.lambda_temporal_tv_loss > 0:
            temoral_tv_loss, l1_sparsity_loss = self.get_temporal_tv_loss(return_sparsity_prior=True)
            loss_dict["temporal_tv_loss"] = self.config.lambda_temporal_tv_loss * temoral_tv_loss.mean()  + \
                                            (self.config.lambda_temporal_tv_loss / 10) * l1_sparsity_loss.mean()

        # import numpy as np
        # out_dir = '/mnt/hdd/debug/famudy_debug2/'
        # os.makedirs(out_dir, exist_ok=True)
        # np.save(out_dir + 'cam_origins.npy', outputs["ray_samples"].frustums.origins.detach().cpu().numpy())
        # np.save(out_dir + 'samples.npy', outputs["ray_samples"].frustums.get_positions().detach().cpu().numpy())
        # np.save(out_dir + 'landmarks.npy', batch["landmarks"].detach().cpu().numpy())
        # assert 1 == 2

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

            metrics_dict["psnr_masked"] = float(psnr_masked)
            metrics_dict["ssim_masked"] = float(ssim_masked)
            metrics_dict["lpips_masked"] = float(lpips_masked)
            metrics_dict["mse_masked"] = float(mse_masked)
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

    def load_state_dict_post_hook(self, module: 'NGPModel', incompatible_keys: _IncompatibleKeys) -> None:
        if '_model.occupancy_grid_eval._binary' in incompatible_keys.missing_keys:
            # Backward compatibility, because occupancy_grid_eval was introduced that defaults to occupancy_grid
            # However, the keys for occupancy_grid_eval are missing in already stored checkpoints even though they
            # are of course just the same as the keys for occupancy_grid
            module.occupancy_grid_eval = module.occupancy_grid
            for missing_key in list(incompatible_keys.missing_keys):
                if missing_key.startswith('_model.occupancy_grid_eval'):
                    incompatible_keys.missing_keys.remove(missing_key)

        if '_model.sampler.occupancy_grid.occupancy_grid._binary' in incompatible_keys.missing_keys:
            # Model was saved in train mode but is loaded in eval mode
            # the sampler attribute holds the state of sampler_train, but sample_eval is requested
            # Can just ignore keys in checkpoint as sampler_eval is already loaded
            module.sampler = module.sampler_eval
            for missing_key in list(incompatible_keys.missing_keys):
                if missing_key.startswith('_model.sampler.occupancy_grid.occupancy_grid'):
                    incompatible_keys.missing_keys.remove(missing_key)

            for unexpected_key in list(incompatible_keys.unexpected_keys):
                if unexpected_key.startswith('_model.sampler.occupancy_grid'):
                    incompatible_keys.unexpected_keys.remove(unexpected_key)

    def fix_view_direction(self, view_direction: torch.Tensor):
        self._fixed_view_direction = view_direction

