"""
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
Adapted from the original implementation to allow configuration of more hyperparams (that were previously hard-coded).
"""
from math import ceil, sqrt
from typing import Optional, List

import torch
from nerfacc import ContractionType, contract
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.hypernerf_field import SE3WarpingField
from nerfstudio.fields.warping import SE3Field, DeformationField
from torch import nn
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
            use_appearance_embedding: bool = False,
            num_images: Optional[int] = None,
            appearance_embedding_dim: int = 32,
            contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
            n_hashgrid_levels: int = 16,
            log2_hashmap_size: int = 19,
            use_spherical_harmonics: bool = True,
            disable_view_dependency: bool = False,
            latent_dim_time: int = 0,
            n_timesteps: int = 1,
            max_ray_samples_chunk_size: int = -1,
            use_deformation_field: bool = False,
            n_freq_pos_warping: int = 7,
            num_layers_deformation_field: int = 3,
            no_hash_encoding: bool = False,
            n_frequencies: int = 12,
            density_threshold: Optional[float] = None,
            use_4d_hashing: bool = float
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type
        self.n_timesteps = n_timesteps
        self.max_ray_samples_chunk_size = max_ray_samples_chunk_size
        self.density_threshold = density_threshold
        self.use_4d_hashing = use_4d_hashing

        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        # TODO: set this properly based on the aabb
        per_level_scale = 1.4472692012786865

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

        if no_hash_encoding:
            hash_grid_encoding_config = {
                "n_dims_to_encode": 3,
                "otype": "Frequency",
                "n_frequencies": n_frequencies
            }
        else:
            hash_grid_encoding_config = {
                "n_dims_to_encode": 4 if use_4d_hashing else 3,
                "otype": "HashGrid",
                "n_levels": n_hashgrid_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            }

        self.deformation_network = None
        if latent_dim_time > 0 and not use_4d_hashing:
            if use_deformation_field:
                # If a deformation field is used, the time embeddings are not fed into the base MLP but into
                # the deformation network instead

                # self.deformation_network = SE3Field(latent_dim_time, hidden_dim)

                # self.deformation_network = DeformationField(num_layers_deformation_field, hidden_dim, latent_dim_time)

                self.deformation_network = SE3WarpingField(
                    n_freq_pos=n_freq_pos_warping,
                    time_embed_dim=latent_dim_time,
                    mlp_num_layers=6,
                    mlp_layer_width=128,
                )

                base_network_encoding_config = hash_grid_encoding_config
                n_base_inputs = 3
            else:
                # Input is [xyz, emb(t)] concatenated
                base_network_encoding_config = {
                    "otype": "Composite",
                    "nested": [
                        hash_grid_encoding_config,
                        {
                            "otype": "Identity"  # Number of remaining input dimensions is automatically derived
                        }
                    ]
                }
                n_base_inputs = 3 + latent_dim_time

            self.time_embedding = nn.Embedding(self.n_timesteps, latent_dim_time)
            init.normal_(self.time_embedding.weight, mean=0., std=0.01 / sqrt(latent_dim_time))
        else:
            base_network_encoding_config = hash_grid_encoding_config
            self.time_embedding = None
            n_base_inputs = 4 if use_4d_hashing else 3

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

        if self.direction_encoding is not None:
            in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        else:
            in_dim = self.geo_feat_dim

        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self,
                    ray_samples: RaySamples,
                    window_deform: Optional[float] = None):

        densities = []
        base_mlp_outs = []

        # Nerfacc's occupancy grid update is quite costl, it queries get_density() with 128^3 which are
        # much more samples than the regular rendering requires
        # To avoid being GPU memory upper-bounded by nerfacc's occupancy grid size, we resort to sequential chunking
        # This should not hurt performance too much as the occupancy grid is only updated periodically
        max_chunk_size = ray_samples.size if self.max_ray_samples_chunk_size == -1 else self.max_ray_samples_chunk_size
        for i_chunk in range(ceil(ray_samples.size / max_chunk_size)):
            ray_samples_chunk = ray_samples[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size]

            positions = ray_samples_chunk.frustums.get_positions()
            positions_flat = positions.view(-1, 3)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

            if self.time_embedding is not None or self.use_4d_hashing:
                timesteps = ray_samples_chunk.timesteps
                if timesteps is None:
                    # Assume ray_samples come from occupancy grid.
                    # We only have one grid to model the whole scene accross time.
                    # Hence, we say, only grid cells that are empty for all timesteps should be really empty.
                    # Thus, we sample random timesteps for these ray samples
                    timesteps = torch.randint(self.n_timesteps, (ray_samples_chunk.size,)).to(positions_flat.device)

                timesteps = timesteps.squeeze(-1)

                if self.use_4d_hashing:
                    timesteps = timesteps.float() / self.n_timesteps
                    base_inputs = [positions_flat, timesteps.unsqueeze(1)]
                else:
                    time_embeddings = self.time_embedding(timesteps)
                    if self.deformation_network is not None:
                        idx_timesteps_deform = timesteps > 0  # Only deform points for other timesteps than canonical

                        if idx_timesteps_deform.any():
                            positions_to_deform = positions_flat[idx_timesteps_deform]

                            # TODO: windows_param scheduling
                            deformed_points = self.deformation_network(positions_to_deform,
                                                                       time_embeddings[idx_timesteps_deform],
                                                                       windows_param=window_deform)

                            # deformed_points = self.deformation_network(positions_to_deform,
                            #                                            time_embeddings[idx_timesteps_deform])

                            positions_flat[idx_timesteps_deform] = deformed_points
                            # positions_flat[idx_timesteps_deform] = positions_to_deform

                            # deformation_inputs = [positions_to_deform, time_embeddings[idx_timesteps_deform]]
                            # deformation_inputs = torch.concat(deformation_inputs, dim=1)
                            # deformation = self.deformation_network(deformation_inputs)
                            # # deformation = torch.zeros_like(deformation)
                            # positions_flat[idx_timesteps_deform] = positions_to_deform + deformation

                        base_inputs = [positions_flat]
                    else:
                        base_inputs = [positions_flat, time_embeddings]
            else:
                base_inputs = [positions_flat]

            base_inputs = torch.concat(base_inputs, dim=1)
            h = self.mlp_base(base_inputs).view(*ray_samples_chunk.frustums.shape, -1)
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
            positions_flat = positions.view(-1, 3)
            positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

            h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, 1 + self.geo_feat_dim)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation.to(positions))
        else:
            density = torch.concat(densities)
            base_mlp_out = torch.concat(base_mlp_outs)

        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        if density_embedding is None:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            if self.direction_encoding is None:
                h = positions.view(-1, 3)
            else:
                d = self.direction_encoding(directions_flat)
                h = torch.cat([d, positions.view(-1, 3)], dim=-1)
        else:
            if self.direction_encoding is None:
                h = density_embedding.view(-1, self.geo_feat_dim)
            else:
                d = self.direction_encoding(directions_flat)
                h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb}

    def forward(self,
                ray_samples: RaySamples,
                compute_normals: bool = False,
                window_deform: Optional[float] = None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples, window_deform=window_deform)
        else:
            density, density_embedding = self.get_density(ray_samples, window_deform=window_deform)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
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

    def get_head_parameters(self) -> List[Parameter]:
        return list(self.mlp_head.parameters())

    def get_base_parameters(self) -> List[Parameter]:
        parameters = []
        for name, params in self.named_parameters():
            if not name.startswith('mlp_head'):
                parameters.append(params)

        return parameters
