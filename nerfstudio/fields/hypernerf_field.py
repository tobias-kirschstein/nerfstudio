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
import torch
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
        time_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.mlp_stem = MLP(
            in_dim=self.position_encoding.get_out_dim() + time_embed_dim,
            out_dim=mlp_layer_width,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )
        self.mlp_r = MLP(
            in_dim=mlp_layer_width,
            out_dim=3,
            num_layers=2,
            layer_width=mlp_layer_width,
        )
        self.mlp_v = MLP(
            in_dim=mlp_layer_width,
            out_dim=3,
            num_layers=2,
            layer_width=mlp_layer_width,
        )

        # diminish the last layer of SE3 Field to approximate an identity transformation
        nn.init.uniform_(self.mlp_r.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.uniform_(self.mlp_v.layers[-1].weight, a=-1e-5, b=1e-5)

    def forward(self, positions, time_embed=None, windows_param=None):
        if time_embed is None:
            return None

        encoded_xyz = self.position_encoding(positions, windows_param=windows_param)  # (R, S, 3)

        feat = self.mlp_stem(torch.cat([encoded_xyz, time_embed], dim=-1))  # (R, S, D)
        r = self.mlp_r(feat).reshape(-1, 3)  # (R*S, 3)
        v = self.mlp_v(feat).reshape(-1, 3)  # (R*S, 3)

        screw_axis = torch.concat([v, r], dim=-1)  # (R*S, 6)
        screw_axis = screw_axis.to(positions.dtype)
        transforms = pytorch3d.transforms.se3_exp_map(screw_axis)
        transformsT = transforms.permute(0, 2, 1)

        p = positions.reshape(-1, 3)

        warped_p = from_homogenous((transformsT @ to_homogenous(p).unsqueeze(-1)).squeeze(-1))
        warped_p = warped_p.to(positions.dtype)

        idx_nan = warped_p.isnan()
        warped_p[idx_nan] = p[idx_nan]  # if deformation is NaN, just use original point

        # Reshape to shape of input positions tensor
        return warped_p.reshape(*positions.shape[:len(positions.shape) - 1], 3)


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
        time_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.mlp_warping = MLP(
            in_dim=self.position_encoding.get_out_dim() + time_embed_dim,
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.normal_(self.mlp_warping.layers[-1].weight, std=1e-4)

    def forward(self, positions, time_embed=None, windows_param=None):
        if time_embed is None:
            return None
        p = self.position_encoding(positions, windows_param=windows_param)
        return self.mlp_warping(torch.cat([p, time_embed], dim=-1))


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
        time_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 64,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1
        )
        self.mlp = MLP(
            in_dim=self.position_encoding.get_out_dim() + time_embed_dim,
            out_dim=out_dim,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.normal_(self.mlp.layers[-1].weight, std=1e-4)

    def forward(self, positions, time_embed=None):
        if time_embed is None:
            return None
        p = self.position_encoding(positions)
        return self.mlp(torch.cat([p, time_embed], dim=-1))


class HyperNeRFField(Field):
    """NeRF Field

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
        extra_dim: int = 0,
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
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_dir, min_freq_exp=0.0, max_freq_exp=n_freq_dir - 1, include_input=True
        )
        if use_hyper_slicing and hyper_slice_dim > 0:
            self.slicing_encoding = WindowedNeRFEncoding(
                in_dim=hyper_slice_dim, num_frequencies=n_freq_slice, min_freq_exp=0.0, max_freq_exp=n_freq_slice - 1
            )
            extra_dim += self.slicing_encoding.get_out_dim()
        else:
            self.slicing_encoding = None

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim() + extra_dim,
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
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
        time_embed: Optional[torch.Tensor] = None,
        warp_field: Optional[SE3WarpingField] = None,
        slice_field: Optional[HyperSlicingField] = None,
        window_alpha: Optional[float] = None,
        window_beta: Optional[float] = None,
    ):
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)

        if warp_field is None and slice_field is None:
            encoded_xyz = self.position_encoding(positions)
            base_inputs = [encoded_xyz, time_embed]
        else:
            base_inputs = []

        if warp_field is not None:
            warped_positions = warp_field(positions, time_embed, window_alpha)

            encoded_xyz = self.position_encoding(warped_positions)
            base_inputs.append(encoded_xyz)
        if slice_field is not None:
            w = slice_field(positions, time_embed)

            encoded_w = self.slicing_encoding(w, windows_param=window_beta)
            base_inputs.append(encoded_w)

        base_inputs = torch.concat(base_inputs, dim=2)
        base_mlp_out = self.mlp_base(base_inputs)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        time_embeddings: Optional[nn.Embedding] = None,
        warp_field: Optional[SE3WarpingField] = None,
        slice_field: Optional[HyperSlicingField] = None,
        window_alpha: Optional[float] = None,
        window_beta: Optional[float] = None,
    ):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if time_embeddings is not None:
            timesteps = ray_samples.timesteps.squeeze(2)  # [R, S]
            time_embed = time_embeddings(timesteps)  # [R, S, D]
        else:
            time_embed = None

        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(
                    ray_samples, time_embed, warp_field, slice_field, window_alpha, window_beta
                )
        else:
            density, density_embedding = self.get_density(
                ray_samples, time_embed, warp_field, slice_field, window_alpha, window_beta
            )

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
