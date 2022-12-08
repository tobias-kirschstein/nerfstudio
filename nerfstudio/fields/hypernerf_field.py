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

"""Classic NeRF field"""
from math import sqrt
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import init
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.engine.generic_scheduler import GenericScheduler
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP, TCNNMLP, TCNNMLPConfig
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

# from nerfstudio.fields.warping import SE3Field, DeformationField
from nerfstudio.fields.deformation_field import DeformationField, SE3Field


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
        n_freq_pos: int = 11,
        n_freq_dir: int = 5,
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        n_timesteps: int = 1,
        time_embed_dim: int = 0,
        use_deformation_field: bool = True,
        n_freq_warp: int = 8,
        alpah_sched: Optional[GenericScheduler] = None,
    ) -> None:
        super().__init__()

        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        # time-conditioning
        assert time_embed_dim > 0
        self.time_embedding = nn.Embedding(n_timesteps, time_embed_dim)
        init.normal_(self.time_embedding.weight, mean=0.0, std=0.01 / sqrt(time_embed_dim))

        if use_deformation_field:
            additional_dim = 0
            # self.deformation_network = DeformationField(
            self.deformation_network = SE3Field(
                n_freq_warp=n_freq_warp, time_embed_dim=time_embed_dim, mlp_num_layers=6, mlp_layer_width=128
            )
        else:
            additional_dim = time_embed_dim
            self.deformation_network = None

        self.alpah_sched = alpah_sched

        # template NeRF
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=n_freq_dir, min_freq_exp=0.0, max_freq_exp=n_freq_dir - 1, include_input=True
        )

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim() + additional_dim,
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

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)

        timesteps = ray_samples.timesteps.squeeze(2)  # [R, S]
        time_embeddings = self.time_embedding(timesteps)  # [R, S, D]

        if self.deformation_network is not None:
            window_alpha = self.alpah_sched.get_value() if self.alpah_sched is not None else None
            positions = self.deformation_network(positions, time_embeddings, window_alpha)

            encoded_xyz = self.position_encoding(positions)
            base_inputs = [encoded_xyz]
        else:
            encoded_xyz = self.position_encoding(positions)
            base_inputs = [encoded_xyz, time_embeddings]

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
