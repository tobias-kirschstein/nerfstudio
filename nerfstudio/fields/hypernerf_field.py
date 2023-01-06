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

"""HyperNeRF field"""
from typing import Any, Dict, Optional, Tuple

import pytorch3d
import pytorch3d.transforms
import tinycudann as tcnn
import torch
from nerfstudio.field_components.hash_encoding import HashEncodingEnsemble, TCNNHashEncodingConfig
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding, WindowedNeRFEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]


class SE3WarpingField(nn.Module):
    """Optimizable temporal deformation using an MLP.
    Args:
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
            self,
            n_freq_pos=7,
            warp_code_dim: int = 8,
            mlp_num_layers: int = 6,
            mlp_layer_width: int = 128,
            skip_connections: Tuple[int] = (4,),
            warp_direction: bool = True,
            use_hash_encoding_ensemble: bool = False,
    ) -> None:
        super().__init__()
        self.warp_direction = warp_direction
        self.use_hash_encoding_ensemble = use_hash_encoding_ensemble

        if use_hash_encoding_ensemble:
            self.position_encoding = HashEncodingEnsemble(warp_code_dim, TCNNHashEncodingConfig())
        else:
            self.position_encoding = WindowedNeRFEncoding(
                in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
            )

        in_dim = self.position_encoding.get_out_dim()
        if not use_hash_encoding_ensemble:
            # Hash encoding ensemble is already conditioned on warp code
            in_dim += warp_code_dim

        self.mlp_stem = MLP(
            in_dim=in_dim,
            out_dim=mlp_layer_width,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.mlp_r = MLP(
            in_dim=mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=mlp_layer_width,
        )
        self.mlp_v = MLP(
            in_dim=mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=mlp_layer_width,
        )

        # diminish the last layer of SE3 Field to approximate an identity transformation
        nn.init.uniform_(self.mlp_r.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.uniform_(self.mlp_v.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.zeros_(self.mlp_r.layers[-1].bias)
        nn.init.zeros_(self.mlp_v.layers[-1].bias)

    def forward(self, positions, directions=None, warp_code=None, windows_param=None, covs=None):
        if warp_code is None:
            return None

        if self.use_hash_encoding_ensemble:
            encoded_xyz = self.position_encoding(
                positions,
                conditioning_code=warp_code,
                windows_param=windows_param
            )
            encoded_xyz = encoded_xyz.to(positions)  # Potentially cast encoded values from Half to Float
            feat = self.mlp_stem(encoded_xyz)  # (R, S, D)
        else:

            encoded_xyz = self.position_encoding(
                positions,
                windows_param=windows_param,
                # covs=covs,
                # not use IPE because the highest freq of PE (2^7) is comparable to the number of samples along a ray (128)
            )  # (R, S, 3)

            feat = self.mlp_stem(torch.cat([encoded_xyz, warp_code], dim=-1))  # (R, S, D)

        r = self.mlp_r(feat).reshape(-1, 3)  # (R*S, 3)
        v = self.mlp_v(feat).reshape(-1, 3)  # (R*S, 3)

        screw_axis = torch.concat([v, r], dim=-1)  # (R*S, 6)
        screw_axis = screw_axis.to(positions.dtype)
        transforms = pytorch3d.transforms.se3_exp_map(screw_axis)
        transforms = transforms.permute(0, 2, 1)
        rots = transforms[:, :3, :3]

        p = positions.reshape(-1, 3)

        warped_p = from_homogenous((transforms @ to_homogenous(p).unsqueeze(-1)).squeeze(-1))
        warped_p = warped_p.to(positions.dtype)

        idx_nan = warped_p.isnan()
        warped_p[idx_nan] = p[idx_nan]  # if deformation is NaN, just use original point

        # Reshape to shape of input positions tensor
        warped_p = warped_p.reshape(*positions.shape[: positions.ndim - 1], 3)

        if self.warp_direction:
            assert directions is not None
            d = directions.reshape(-1, 3)

            warped_d = (rots @ d.unsqueeze(-1)).squeeze(-1)
            warped_d = warped_d.to(directions.dtype)

            idx_nan = warped_d.isnan()
            warped_d[idx_nan] = d[idx_nan]  # if deformation is NaN, just use original point

            warped_d = warped_d.reshape(
                *directions.shape[: directions.ndim - 1], 3
            )  # .detach()  #TODO: further test its inluece to training stability
        else:
            warped_d = directions

        if covs is not None:
            c = covs.reshape(-1, *covs.shape[-2:])

            warped_c = rots @ c @ rots.transpose(-2, -1)
            warped_c = warped_c.to(covs.dtype)

            idx_nan = warped_c.isnan()
            warped_c[idx_nan] = c[idx_nan]  # if deformation is NaN, just use original point

            warped_c = warped_c.reshape(
                covs.shape
            ).detach()  # detach warped_c to avoid back-propagating to rot (necessary to avoid NaN)

            return warped_p, warped_d, warped_c
        else:
            return warped_p, warped_d


class DeformationField(nn.Module):
    """Optimizable temporal deformation using an MLP.
    Args:
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
            self,
            n_freq_pos=7,
            warp_code_dim: int = 8,
            mlp_num_layers: int = 6,
            mlp_layer_width: int = 128,
            skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.mlp_warping = MLP(
            in_dim=self.position_encoding.get_out_dim() + warp_code_dim,
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.normal_(self.mlp_warping.layers[-1].weight, std=1e-4)

    def forward(self, positions, warp_code=None, windows_param=None):
        if warp_code is None:
            return None
        p = self.position_encoding(positions, windows_param=windows_param)
        return self.mlp_warping(torch.cat([p, warp_code], dim=-1))


class HyperSlicingField(nn.Module):
    """Optimizable hyper slicing using an MLP.
    Args:
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
            self,
            n_freq_pos: int = 7,
            out_dim: int = 2,
            warp_code_dim: int = 8,
            mlp_num_layers: int = 6,
            mlp_layer_width: int = 64,
            skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()

        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=False
        )

        # self.position_encoding = NeRFEncoding(
        #     in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1
        # )
        self.mlp = MLP(
            in_dim=self.position_encoding.get_out_dim() + warp_code_dim,
            out_dim=out_dim,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.normal_(self.mlp.layers[-1].weight, std=1e-5)

    def forward(self, positions, warp_code=None, covs=None, window_param: Optional[float] = None):
        if warp_code is None:
            return None
        p = self.position_encoding(positions, windows_param=window_param, covs=covs)

        return self.mlp(torch.cat([p, warp_code], dim=-1))


class HyperNeRFField(Field):
    """HyperNeRF Field

    Args:
        n_freq_pos: Number of frequencies for position in positional encoding.
        n_freq_dir: Number of frequencies for direction in positional encoding..
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for ourput head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
            self,
            n_freq_pos: int = 9,
            n_freq_dir: int = 5,
            use_hyper_slicing: bool = True,
            n_freq_slice: int = 2,
            hyper_slice_dim: int = 2,
            base_extra_dim: int = 0,
            head_extra_dim: int = 0,
            base_mlp_num_layers: int = 8,
            base_mlp_layer_width: int = 256,
            head_mlp_num_layers: int = 2,
            head_mlp_layer_width: int = 128,
            skip_connections: Tuple[int] = (4,),
            field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
            use_integrated_encoding: bool = False,
            spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        # template NeRF
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_dir, min_freq_exp=0.0, max_freq_exp=n_freq_dir - 1, include_input=True
        )
        if use_hyper_slicing and hyper_slice_dim > 0:
            self.slicing_encoding = WindowedNeRFEncoding(
                in_dim=hyper_slice_dim, num_frequencies=n_freq_slice, min_freq_exp=0.0, max_freq_exp=n_freq_slice - 1
            )
            base_extra_dim += self.slicing_encoding.get_out_dim()
        else:
            self.slicing_encoding = None

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim() + base_extra_dim,
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim() + head_extra_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(
            self,
            ray_samples: RaySamples,
            warp_code: Optional[torch.Tensor] = None,
            warp_field: Optional[SE3WarpingField] = None,
            slice_field: Optional[HyperSlicingField] = None,
            window_alpha: Optional[float] = None,
            window_beta: Optional[float] = None,
            window_gamma: Optional[float] = None,
    ):
        warped_directions = None
        base_inputs = []
        directions = ray_samples.frustums.directions

        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)

            if warp_field is None and slice_field is None:
                encoded_xyz = self.position_encoding(
                    gaussian_samples.mean, windows_param=window_gamma, covs=gaussian_samples.cov
                )
                base_inputs.append(encoded_xyz)
                if warp_code is not None:
                    base_inputs.append(warp_code)

            if warp_field is not None:
                warped_positions, warped_directions, warped_cov = warp_field(
                    gaussian_samples.mean,
                    directions,
                    warp_code,
                    window_alpha,
                    covs=gaussian_samples.cov,
                )

                encoded_xyz = self.position_encoding(warped_positions, windows_param=window_gamma, covs=warped_cov)
                base_inputs.append(encoded_xyz)
            if slice_field is not None:
                w = slice_field(
                    gaussian_samples.mean,
                    warp_code,
                    # covs=gaussian_samples.cov,
                    # not use IPE because the highest freq of PE (2^7) is comparable to the number of samples along a ray (128)
                )
                assert self.slicing_encoding is not None
                encoded_w = self.slicing_encoding(w, windows_param=window_beta)
                base_inputs.append(encoded_w)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)

            if warp_field is None and slice_field is None:
                encoded_xyz = self.position_encoding(positions, windows_param=window_gamma)
                base_inputs.append(encoded_xyz)
                if warp_code is not None:
                    base_inputs.append(warp_code)

            if warp_field is not None:
                warped_positions, warped_directions = warp_field(positions, directions, warp_code, window_alpha)

                encoded_xyz = self.position_encoding(warped_positions, windows_param=window_gamma)
                base_inputs.append(encoded_xyz)
            if slice_field is not None:
                w = slice_field(positions, warp_code)

                assert self.slicing_encoding is not None
                encoded_w = self.slicing_encoding(w, windows_param=window_beta)
                base_inputs.append(encoded_w)

        base_inputs = torch.concat(base_inputs, dim=-1)
        base_mlp_out = self.mlp_base(base_inputs)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out, warped_directions

    def get_outputs(
            self,
            ray_samples: RaySamples,
            warped_direction: Optional[TensorType] = None,
            density_embedding: Optional[TensorType] = None,
            appearance_code: Optional[torch.Tensor] = None,
            camera_code: Optional[torch.Tensor] = None,
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            if warped_direction is not None:
                encoded_dir = self.direction_encoding(warped_direction)
            else:
                encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            head_inputs = [encoded_dir, density_embedding]

            if appearance_code is not None:
                head_inputs.append(appearance_code)
            if camera_code is not None:
                head_inputs.append(camera_code)

            mlp_out = self.mlp_head(torch.cat(head_inputs, dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

    def forward(
            self,
            ray_samples: RaySamples,
            compute_normals: bool = False,
            warp_embeddings: Optional[nn.Embedding] = None,
            appearance_embeddings: Optional[nn.Embedding] = None,
            camera_embeddings: Optional[nn.Embedding] = None,
            warp_field: Optional[SE3WarpingField] = None,
            slice_field: Optional[HyperSlicingField] = None,
            window_alpha: Optional[float] = None,
            window_beta: Optional[float] = None,
            window_gamma: Optional[float] = None,
    ):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if warp_embeddings is not None:
            timesteps = ray_samples.timesteps.squeeze(2)  # [R, S]
            warp_code = warp_embeddings(timesteps)  # [R, S, D]
        else:
            warp_code = None

        if appearance_embeddings is not None:
            timesteps = ray_samples.timesteps.squeeze(2)  # [R, S]
            appearance_code = appearance_embeddings(timesteps)  # [R, S, D]
        else:
            appearance_code = None

        if camera_embeddings is not None:
            if self.training:
                camera_indices = ray_samples.camera_indices.squeeze(2)  # [R, S]
                camera_code = camera_embeddings(camera_indices)  # [R, S, D]
            else:
                camera_code = camera_embeddings.weight.mean(0)[None, None, :].repeat(*ray_samples.shape, 1)
        else:
            camera_code = None

        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, warped_directions = self.get_density(
                    ray_samples, warp_code, warp_field, slice_field, window_alpha, window_beta, window_gamma
                )
        else:
            density, density_embedding, warped_directions = self.get_density(
                ray_samples, warp_code, warp_field, slice_field, window_alpha, window_beta, window_gamma
            )

        field_outputs = self.get_outputs(
            ray_samples,
            warped_directions,
            density_embedding=density_embedding,
            appearance_code=appearance_code,
            camera_code=camera_code,
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


class HashHyperNeRFField(HyperNeRFField):
    """HyperNeRF Field with HashEncoding

    Args:
        n_freq_pos: Number of frequencies for position in positional encoding.
        n_freq_dir: Number of frequencies for direction in positional encoding..
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for ourput head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
            self,
            aabb: TensorType,
            use_hyper_slicing: bool = True,
            n_freq_slice: int = 2,
            hyper_slice_dim: int = 2,
            base_extra_dim: int = 0,
            head_extra_dim: int = 0,
            base_mlp_num_layers: int = 2,
            base_mlp_layer_width: int = 64,
            head_mlp_num_layers: int = 2,
            head_mlp_layer_width: int = 128,
            field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
    ) -> None:
        super(HyperNeRFField, self).__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)

        # template NeRF
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 1,
            },
        )
        if use_hyper_slicing and hyper_slice_dim > 0:
            self.slicing_encoding = WindowedNeRFEncoding(
                in_dim=hyper_slice_dim, num_frequencies=n_freq_slice, min_freq_exp=0.0, max_freq_exp=n_freq_slice - 1
            )
            base_extra_dim += self.slicing_encoding.get_out_dim()
        else:
            self.slicing_encoding = None

        geo_feat_dim = 15
        base_in_dim = 3
        n_hashgrid_levels = 16
        log2_hashmap_size = 19
        per_level_hashgrid_scale = 1.4472692012786865

        hash_grid_encoding_config = {
            "n_dims_to_encode": base_in_dim,
            "otype": "HashGrid",
            "n_levels": n_hashgrid_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": 16,
            "per_level_scale": per_level_hashgrid_scale,
        }

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=base_in_dim,
            n_output_dims=geo_feat_dim,
            encoding_config=hash_grid_encoding_config,
            network_config={
                "otype": "FullyFusedMLP" if base_mlp_layer_width <= 128 else "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": base_mlp_layer_width,
                "n_hidden_layers": base_mlp_num_layers - 1,
            },
        )

        head_in_dim = geo_feat_dim
        if self.direction_encoding is not None:
            head_in_dim += self.direction_encoding.n_output_dims

        self.mlp_head = MLP(
            in_dim=head_in_dim + head_extra_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.n_output_dims)
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(
            self,
            ray_samples: RaySamples,
            warp_code: Optional[torch.Tensor] = None,
            warp_field: Optional[SE3WarpingField] = None,
            slice_field: Optional[HyperSlicingField] = None,
            window_alpha: Optional[float] = None,
            window_beta: Optional[float] = None,
    ):
        warped_directions = None

        positions = ray_samples.frustums.get_positions()
        directions = ray_samples.frustums.directions

        if warp_field is None and slice_field is None:
            encoded_xyz = self.position_encoding(positions)
            base_inputs = [encoded_xyz]
            if warp_code is not None:
                base_inputs.append(warp_code)
        else:
            base_inputs = []

        if warp_field is not None:
            warped_positions, warped_directions = warp_field(positions, directions, warp_code, window_alpha)

            warped_positions = warped_positions.view(-1, 3)
            warped_positions = (warped_positions - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

            warped_directions = warped_directions.view(-1, 3)

            base_inputs.append(warped_positions)
        if slice_field is not None:
            w = slice_field(positions, warp_code)

            assert self.slicing_encoding is not None
            encoded_w = self.slicing_encoding(w, windows_param=window_beta)
            base_inputs.append(encoded_w)

        base_inputs = torch.concat(base_inputs, dim=-1)
        base_mlp_out = self.mlp_base(base_inputs).to(base_inputs)
        density = self.field_output_density(base_mlp_out).reshape([*positions.shape[:2], -1])
        return density, base_mlp_out, warped_directions

    def get_outputs(
            self,
            ray_samples: RaySamples,
            warped_direction: Optional[TensorType] = None,
            density_embedding: Optional[TensorType] = None,
            appearance_code: Optional[torch.Tensor] = None,
            camera_code: Optional[torch.Tensor] = None,
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            if warped_direction is not None:
                encoded_dir = self.direction_encoding(warped_direction)
            else:
                encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            head_inputs = [encoded_dir, density_embedding]

            if appearance_code is not None:
                head_inputs.append(appearance_code)
            if camera_code is not None:
                head_inputs.append(camera_code.view(-1, camera_code.shape[-1]))

            mlp_out = self.mlp_head(torch.cat(head_inputs, dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out).reshape([*ray_samples.shape, -1])
        return outputs
