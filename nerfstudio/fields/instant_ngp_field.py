"""
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
Adapted from the original implementation to allow configuration of more hyperparams (that were previously hard-coded).
"""
from collections import defaultdict
from math import ceil
from typing import List, Optional, Callable, Tuple, Dict, Literal

import torch
from nerfacc import ContractionType, contract
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components import MLP
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.hash_encoding import HashEncodingEnsemble, TCNNHashEncodingConfig, \
    HashEnsembleMixingType, BlendFieldConfig, MultiDeformConfig, MultiDeformSE3Config
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.torch import disable_gradients_for
from torch.nn import init
from torch.nn.parameter import Parameter
from torchtyping import TensorType

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class TCNNInstantNGPField(Field):
    """TCNN implementation of the Instant-NGP field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        num_images: number of images, requried if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
    """

    def __init__(
            self,
            aabb,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            num_images: Optional[int] = None,
            use_appearance_embedding: bool = False,
            appearance_embedding_dim: int = 32,
            use_camera_embedding: bool = False,
            camera_embedding_dim: int = 8,
            use_affine_color_transformation: bool = False,
            contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
            n_hashgrid_levels: int = 16,
            log2_hashmap_size: int = 19,
            per_level_hashgrid_scale: float = 1.4472692012786865,
            hashgrid_base_resolution: int = 16,
            hashgrid_n_features_per_level: int = 2,
            use_spherical_harmonics: bool = True,
            disable_view_dependency: bool = False,
            latent_dim_time: int = 0,
            n_timesteps: int = 1,
            max_ray_samples_chunk_size: int = -1,
            fix_canonical_space: bool = False,
            n_freq_pos_ambient: int = 7,
            timestep_canonical: Optional[int] = 0,
            use_time_conditioning_for_base_mlp: bool = False,
            use_time_conditioning_for_rgb_mlp: bool = False,
            use_deformation_skip_connection: bool = False,
            use_smoothstep_hashgrid_interpolation: bool = False,
            n_ambient_dimensions: int = 0,

            use_hash_encoding_ensemble: bool = False,
            hash_encoding_ensemble_n_levels: int = 16,
            hash_encoding_ensemble_features_per_level: int = 2,
            hash_encoding_ensemble_n_tables: Optional[int] = None,
            hash_encoding_ensemble_mixing_type: HashEnsembleMixingType = 'blend',
            hash_encoding_ensemble_n_heads: Optional[int] = None,
            hash_encoding_ensemble_disable_initial: bool = False,
            only_render_hash_table: Optional[int] = None,
            n_freq_pos_warping: int = 7,

            # only used when mixing_type == 'mlp_blend_field'
            blend_field_hidden_dim: int = 64,
            blend_field_n_layers: int = 4,
            blend_field_out_activation: Optional[Literal['Tanh', 'Normalization']] = None,
            blend_field_n_freq_enc: int = 0,
            blend_field_skip_connections: Optional[Tuple[int]] =  None,

            no_hash_encoding: bool = False,
            n_frequencies: int = 12,
            density_threshold: Optional[float] = None,
            use_4d_hashing: bool = False,

            density_fn_ray_samples_transform: Callable[
                [RaySamples], Tuple[RaySamples, Optional[torch.TensorType]]] = lambda x: x
    ) -> None:
        super().__init__(density_fn_ray_samples_transform=density_fn_ray_samples_transform)

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type
        self.n_timesteps = n_timesteps
        self.max_ray_samples_chunk_size = max_ray_samples_chunk_size
        self.density_threshold = density_threshold
        self.use_4d_hashing = use_4d_hashing
        self.n_ambient_dimensions = n_ambient_dimensions
        self.fix_canonical_space = fix_canonical_space
        self.timestep_canonical = timestep_canonical
        self.use_time_conditioning_for_base_mlp = use_time_conditioning_for_base_mlp
        self.use_time_conditioning_for_rgb_mlp = use_time_conditioning_for_rgb_mlp
        self.use_deformation_skip_connection = use_deformation_skip_connection
        self.use_hash_encoding_ensemble = use_hash_encoding_ensemble

        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        self.use_camera_embedding = use_camera_embedding
        self.use_affine_color_transformation = use_affine_color_transformation
        if use_camera_embedding:
            assert num_images is not None
            assert n_timesteps is not None, "Currently, camera embedding assumes hat cameras don't move. I.e., there is only 1 camera embedding per camera"
            self.camera_embedding = Embedding(int(num_images / n_timesteps), camera_embedding_dim)
            init.uniform_(self.camera_embedding.embedding.weight, a=-0.05, b=0.05)

            if use_affine_color_transformation:
                self.affine_color_transformation = MLP(camera_embedding_dim, 3, 64, 9 + 3)

        # ----------------------------------------------------------
        # Base network with hash encoding
        # ----------------------------------------------------------

        if no_hash_encoding:
            hash_grid_encoding_config = {
                "n_dims_to_encode": 3,
                "otype": "Frequency",
                "n_frequencies": n_frequencies
            }
        elif use_hash_encoding_ensemble:
            n_hashtables = latent_dim_time if hash_encoding_ensemble_mixing_type == 'blend' else hash_encoding_ensemble_n_tables
            self.hash_encoding_ensemble = HashEncodingEnsemble(
                n_hashtables,
                TCNNHashEncodingConfig(n_levels=hash_encoding_ensemble_n_levels,
                                       n_features_per_level=hash_encoding_ensemble_features_per_level),
                mixing_type=hash_encoding_ensemble_mixing_type,
                dim_conditioning_code=latent_dim_time,
                n_heads=hash_encoding_ensemble_n_heads,
                only_render_hash_table=only_render_hash_table,
                blend_field_config=BlendFieldConfig(n_hidden_dims=blend_field_hidden_dim,
                                                    n_layers=blend_field_n_layers,
                                                    output_activation=blend_field_out_activation,
                                                    n_freq_pos_enc=blend_field_n_freq_enc,
                                                    skip_connections=blend_field_skip_connections
                                                    ) if hash_encoding_ensemble_mixing_type in {'mlp_blend_field', 'multi_deform_blend', 'multi_deform_blend++'} else None,
                multi_deform_config=MultiDeformConfig(n_hidden_dims=blend_field_hidden_dim, # TODO: for now sharing hyperparams wih blend field
                                                      n_layers=blend_field_n_layers,
                                                      n_freq_pos_enc=blend_field_n_freq_enc,
                ) if hash_encoding_ensemble_mixing_type in ['multi_deform_blend', 'multi_deform_blend++'] else None,
                multi_deform_se3_config=MultiDeformSE3Config(
                    n_freq_pos_enc=n_freq_pos_warping
                ) if hash_encoding_ensemble_mixing_type in ['multi_deform_blend', 'multi_deform_blend++'] else None,
                disable_initial_hash_ensemble=hash_encoding_ensemble_disable_initial
            )

            # Hash encoding is computed seperately, so base MLP just takes inputs without adding encoding
            hash_grid_encoding_config = {
                "otype": "Identity",
                "n_dims_to_encode": self.hash_encoding_ensemble.get_out_dim(),
            }
        else:
            hash_grid_encoding_config = {
                "n_dims_to_encode": 4 if use_4d_hashing else 3,
                "otype": "HashGrid",
                "n_levels": n_hashgrid_levels,
                "n_features_per_level": hashgrid_n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": hashgrid_base_resolution,
                "per_level_scale": per_level_hashgrid_scale,
                "interpolation": "Smoothstep" if use_smoothstep_hashgrid_interpolation else "Linear"
            }

        base_network_encoding_config = hash_grid_encoding_config
        if use_4d_hashing:
            n_base_inputs = 4
        elif use_hash_encoding_ensemble:
            n_base_inputs = self.hash_encoding_ensemble.get_out_dim()
        else:
            n_base_inputs = 3

        if n_ambient_dimensions > 0:
            n_base_inputs += n_ambient_dimensions
            base_network_encoding_config = {
                "otype": "Composite",
                "nested": [
                    base_network_encoding_config,
                    {
                        "otype": "Frequency",
                        "n_dims_to_encode": n_ambient_dimensions,
                        "n_frequencies": n_freq_pos_ambient
                    }
                ]
            }

        if use_time_conditioning_for_base_mlp:
            # TODO: This cannot go together with 4D canonical space
            base_network_encoding_config = {
                "otype": "Composite",
                "nested": [
                    base_network_encoding_config,
                    {
                        "otype": "Identity",  # Number of remaining input dimensions is automatically derived
                    }
                ]
            }
            n_base_inputs += latent_dim_time

        if use_deformation_skip_connection:
            if not base_network_encoding_config['otype'] == 'Composite':
                base_network_encoding_config = {
                    "otype": "Composite",
                    "nested": [
                        base_network_encoding_config,
                        {
                            "otype": "Identity",  # Number of remaining input dimensions is automatically derived
                        }
                    ]
                }
            n_base_inputs += 3 * 16  # For some reason the skip connection only works with a higher dimensionality

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_base_inputs,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config=base_network_encoding_config,
            network_config={
                "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # ----------------------------------------------------------
        # RGB network
        # ----------------------------------------------------------

        if disable_view_dependency:
            self.direction_encoding = None
        elif use_spherical_harmonics:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                }
            )
        else:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Identity"
                },
            )

        if self.direction_encoding is not None:
            in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        else:
            in_dim = self.geo_feat_dim

        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim

        if self.use_camera_embedding and not self.use_affine_color_transformation:
            # If affine color transformation is used, we do not condition the RGB network on the camera embedding
            in_dim += self.camera_embedding.out_dim

        if use_time_conditioning_for_rgb_mlp:
            in_dim += latent_dim_time

        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None" if use_camera_embedding and use_affine_color_transformation else "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self,
                    ray_samples: RaySamples,
                    window_canonical: Optional[float] = None,
                    window_blend: Optional[float] = None,
                    window_hash_tables: Optional[float] = None,
                    window_deform: Optional[float] = None,
                    time_codes: Optional[torch.Tensor] = None):

        densities = []
        base_mlp_outs = []

        if self.n_ambient_dimensions > 0 and ray_samples.frustums.ambient_coordinates is None:
            # Assume that this is a forward pass for the occupancy grid
            # Randomly sample ambient coordinates in [-1, 1]
            ambient_coordinates = torch.rand((*ray_samples.frustums.shape, self.n_ambient_dimensions))
            ambient_coordinates = ambient_coordinates.to(ray_samples.frustums.origins) * 2 - 1
            ray_samples.frustums.ambient_coordinates = ambient_coordinates

        # Nerfacc's occupancy grid update is quite costl, it queries get_density() with 128^3 which are
        # much more samples than the regular rendering requires
        # To avoid being GPU memory upper-bounded by nerfacc's occupancy grid size, we resort to sequential chunking
        # This should not hurt performance too much as the occupancy grid is only updated periodically
        max_chunk_size = ray_samples.size if self.max_ray_samples_chunk_size == -1 else self.max_ray_samples_chunk_size
        positions = ray_samples.frustums.get_positions()
        d_spatial = positions.shape[-1]
        timesteps = None if ray_samples.timesteps is None else ray_samples.timesteps.squeeze(-1)

        for i_chunk in range(ceil(ray_samples.size / max_chunk_size)):
            positions_chunk = positions[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size]
            positions_flat = positions_chunk.view(-1, d_spatial)
            # positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)
            # Manually compute contraction here, as contract(..) is not differentiable wrt the position input
            # Do not contract ambient dimensions (i.e., only the first 3 dimensions)
            #positions_flat[:, :3] = SceneBox.get_normalized_positions(positions_flat[:, :3], self.aabb)
            if self.n_ambient_dimensions > 0:
                positions_flat = torch.stack([ (positions_flat[:, :3] - self.aabb[0]) / (self.aabb[1] - self.aabb[0]),
                                               positions_flat[:, 3:]], dim=-1)
            else:
                positions_flat = (positions_flat - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

            timesteps_chunk = None
            if timesteps is not None:
                timesteps_chunk = timesteps[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size]

            if self.use_4d_hashing:
                if timesteps is None:
                    # Assume ray_samples come from occupancy grid.
                    # We only have one grid to model the whole scene accross time.
                    # Hence, we say, only grid cells that are empty for all timesteps should be really empty.
                    # Thus, we sample random timesteps for these ray samples
                    timesteps_chunk = torch.randint(self.n_timesteps, (len(timesteps_chunk),)).to(positions_flat.device)

                if self.use_4d_hashing:
                    timesteps_chunk = timesteps_chunk.float() / self.n_timesteps
                    base_inputs = [positions_flat, timesteps_chunk.unsqueeze(1)]
            else:
                if self.use_hash_encoding_ensemble:
                    time_codes_chunk = time_codes[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size]
                    embeddings = self.hash_encoding_ensemble(positions_flat,
                                                             conditioning_code=time_codes_chunk,
                                                             windows_param=window_canonical,
                                                             windows_param_blend_field=window_blend,
                                                             windows_param_tables=window_hash_tables,
                                                             windows_param_deform=window_deform,
                                                             )
                    base_inputs = [embeddings]
                else:
                    base_inputs = [positions_flat]

                if self.use_time_conditioning_for_base_mlp:
                    assert time_codes is not None, "If use_time_conditioning_for_base_mlp is set, time_codes have to be provided"

                    base_inputs.append(time_codes[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size])

                if self.use_deformation_skip_connection:
                    base_inputs.append(positions_flat)
                    # For some reason the skip connection only works if we increase the dimensions.
                    # Hence, we feed a bunch of zeros into the base MLP
                    base_inputs.append(torch.zeros_like(positions_flat[:, [0 for _ in range(3 * 15)]]))

            base_inputs = torch.concat(base_inputs, dim=1)
            if timesteps_chunk is not None and self.fix_canonical_space:
                # TODO: Experimental
                # Only accumulate gradients for mlp_base for canonical space rays.
                # All other timesteps should only update the deformation field
                assert self.timestep_canonical is not None
                idx_timesteps_deform = timesteps_chunk != self.timestep_canonical

                h_canonical = self.mlp_base(base_inputs[~idx_timesteps_deform])

                h = torch.zeros((base_inputs.shape[0], *h_canonical.shape[1:]),
                                dtype=h_canonical.dtype,
                                device=h_canonical.device)

                if (~idx_timesteps_deform).any():
                    h[~idx_timesteps_deform] = h_canonical

                if idx_timesteps_deform.any():
                    with disable_gradients_for(self.mlp_base):
                        h_deform = self.mlp_base(base_inputs[idx_timesteps_deform])
                    h[idx_timesteps_deform] = h_deform

                h = h.view(*positions_chunk.shape[:-1], -1)
            else:
                h = self.mlp_base(base_inputs).view(*positions_chunk.shape[:-1], -1)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            density = trunc_exp(density_before_activation.to(positions))

            if not self.training and self.density_threshold is not None:
                density[density < self.density_threshold] = 0

            densities.append(density)
            base_mlp_outs.append(base_mlp_out)

        if ray_samples.size == 0:
            # Weird edge case
            positions = ray_samples.frustums.get_positions()
            positions_flat = positions.view(-1, d_spatial)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

            h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, 1 + self.geo_feat_dim)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation.to(positions))
        else:
            density = torch.concat(densities)
            base_mlp_out = torch.concat(base_mlp_outs)

        return density, base_mlp_out

    def get_outputs(self,
                    ray_samples: RaySamples,
                    density_embedding: Optional[TensorType] = None,
                    time_codes: Optional[TensorType] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        if density_embedding is None:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            d_spatial = positions.shape[-1]
            if self.direction_encoding is None:
                h = positions.view(-1, d_spatial)
            else:
                d = self.direction_encoding(directions_flat)
                h = torch.cat([d, positions.view(-1, d_spatial)], dim=-1)
        else:
            if self.direction_encoding is None:
                h = density_embedding.view(-1, self.geo_feat_dim)
            else:
                d = self.direction_encoding(directions_flat)
                h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        # Appearance embeddings
        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze(dim=-1)
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        # Color correction with camera embeddings
        predicted_color_transformation = None
        if self.use_camera_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze(dim=-1)
            if self.training:
                camera_code = self.camera_embedding(camera_indices)
            else:
                # During evaluation we use the mean over all camera embeddings
                camera_code = self.camera_embedding.embedding.weight.mean(0)[None, :].repeat(*ray_samples.shape, 1)

            if self.use_affine_color_transformation:
                predicted_color_transformation = self.affine_color_transformation(camera_code)
            else:
                # Do not condition on camera embedding when affine color transformations are used
                h = torch.cat([h, camera_code.view(-1, self.camera_embedding.out_dim)], dim=-1)

        if self.use_time_conditioning_for_rgb_mlp:
            assert time_codes is not None, "If use_time_conditioning_for_rgb_mlp is set, time_codes have to be provided"
            h = torch.cat([h, time_codes], dim=-1)

        # RGB MLP
        if ray_samples.timesteps is not None and self.fix_canonical_space:
            # Ensure that only canonical space rays accumulate gradients
            assert self.timestep_canonical is not None
            idx_timesteps_deform = ray_samples.timesteps.squeeze(-1) != self.timestep_canonical

            rgb_canonical = self.mlp_head(h[~idx_timesteps_deform])

            rgb = torch.zeros((h.shape[0], *rgb_canonical.shape[1:]),
                              dtype=rgb_canonical.dtype,
                              device=rgb_canonical.device)

            if (~idx_timesteps_deform).any():
                rgb[~idx_timesteps_deform] = rgb_canonical

            if idx_timesteps_deform.any():
                with disable_gradients_for(self.mlp_head):
                    rgb_deform = self.mlp_head(h[idx_timesteps_deform])
                rgb[idx_timesteps_deform] = rgb_deform

            rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        else:
            rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)

        if self.use_camera_embedding and self.use_affine_color_transformation:
            color_matrix = predicted_color_transformation[:, :9].reshape(-1, 3, 3)  # [B, 3, 3]
            color_offset = predicted_color_transformation[:, 9:]  # [B, 3]
            rgb = rgb.unsqueeze(2)  # [B, 3, 1]
            rgb = torch.bmm(color_matrix, rgb)  # [B, 3, 1]
            rgb = rgb.squeeze(2)  # [B, 3]
            rgb = rgb + color_offset
            rgb = rgb.sigmoid()

        return {FieldHeadNames.RGB: rgb}

    def forward(self,
                ray_samples: RaySamples,
                compute_normals: bool = False,
                window_canonical: Optional[float] = None,
                window_blend: Optional[float] = None,
                window_hash_tables: Optional[float] = None,
                window_deform: Optional[float] = None,
                time_codes: Optional[TensorType] = None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples,
                                                              window_canonical=window_canonical,
                                                              window_blend=window_blend,
                                                              window_hash_tables=window_hash_tables,
                                                              window_deform=window_deform,
                                                              time_codes=time_codes)
        else:
            density, density_embedding = self.get_density(ray_samples,
                                                          window_canonical=window_canonical,
                                                          window_blend=window_blend,
                                                          window_hash_tables=window_hash_tables,
                                                          window_deform=window_deform,
                                                          time_codes=time_codes)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding, time_codes=time_codes)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

    def get_opacity(self, positions: TensorType["bs":..., 3], step_size) -> TensorType["bs":..., 1]:
        """Returns the opacity for a position. Used primarily by the occupancy grid.

        Args:
            positions: the positions to evaluate the opacity at.
            step_size: the step size to use for the opacity evaluation.
        """
        density = self.density_fn(positions)
        ## TODO: We should scale step size based on the distortion. Currently it uses too much memory.
        # aabb_min, aabb_max = self.aabb[0], self.aabb[1]
        # if self.contraction_type is not ContractionType.AABB:
        #     x = (positions - aabb_min) / (aabb_max - aabb_min)
        #     x = x * 2 - 1  # aabb is at [-1, 1]
        #     mag = x.norm(dim=-1, keepdim=True)
        #     mask = mag.squeeze(-1) > 1

        #     dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (1 / mag**3 - (2 * mag - 1) / mag**4)
        #     dev[~mask] = 1.0
        #     dev = torch.clamp(dev, min=1e-6)
        #     step_size = step_size / dev.norm(dim=-1, keepdim=True)
        # else:
        #     step_size = step_size * (aabb_max - aabb_min)

        opacity = density * step_size
        return opacity

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = defaultdict(list)

        param_groups["fields"].extend(self.mlp_base.parameters())
        param_groups["fields"].extend(self.mlp_head.parameters())

        if self.use_hash_encoding_ensemble:
            param_groups_hash_ensemble = self.hash_encoding_ensemble.get_param_groups()
            for name, param_group in param_groups_hash_ensemble.items():
                param_groups[name].extend(param_group)

        if self.use_camera_embedding:
            param_groups["embeddings"].extend(self.camera_embedding.parameters())

        if self.use_appearance_embedding:
            param_groups["embeddings"].extend(self.appearance_embedding.parameters())

            if self.use_affine_color_transformation:
                param_groups["fields"].extend(self.affine_color_transformation.parameters())

        return param_groups

    def get_head_parameters(self) -> List[Parameter]:
        return list(self.mlp_head.parameters())

    def get_base_parameters(self) -> List[Parameter]:
        parameters = []
        for name, params in self.named_parameters():
            if not name.startswith('mlp_head'):
                parameters.append(params)

        return parameters
