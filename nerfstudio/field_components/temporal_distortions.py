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

"""Space distortions which occur as a function of time."""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.hypernerf_field import SE3WarpingField
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP


class TemporalDistortion(nn.Module):
    """Apply spatial distortions as a function of time"""

    def forward(self, positions: TensorType["bs":..., 3], times: Optional[TensorType[1]]) -> TensorType["bs":..., 3]:
        """
        Args:
            positions: Samples to translate as a function of time
            times: times for each sample

        Returns:
            Translated positions.
        """


class TemporalDistortionKind(Enum):
    """Possible temporal distortion names"""

    DNERF = "dnerf"

    def to_temporal_distortion(self, config: Dict[str, Any]) -> TemporalDistortion:
        """Converts this kind to a temporal distortion"""
        if self == TemporalDistortionKind.DNERF:
            return DNeRFDistortion(**config)
        raise NotImplementedError(f"Unknown temporal distortion kind {self}")


class DNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
            self,
            position_encoding: Encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
            ),
            temporal_encoding: Encoding = NeRFEncoding(
                in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
            ),
            mlp_num_layers: int = 4,
            mlp_layer_width: int = 256,
            skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.mlp_deform = MLP(
            in_dim=self.position_encoding.get_out_dim() + self.temporal_encoding.get_out_dim(),
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(self, positions, times=None):
        if times is None:
            return None
        p = self.position_encoding(positions)
        t = self.temporal_encoding(times)
        return self.mlp_deform(torch.cat([p, t], dim=-1))


class SE3Distortion(nn.Module):

    def __init__(self,
                 n_freq_pos=7,
                 warp_code_dim: int = 8,
                 mlp_num_layers: int = 6,
                 mlp_layer_width: int = 128,
                 skip_connections: Tuple[int] = (4,),
                 warp_direction: bool = True,
                 ):
        super(SE3Distortion, self).__init__()

        self.se3_field = SE3WarpingField(
            n_freq_pos=n_freq_pos,
            warp_code_dim=warp_code_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_layer_width=mlp_layer_width,
            skip_connections=skip_connections,
            warp_direction=warp_direction
        )

    def forward(self, ray_samples: RaySamples, warp_code=None, windows_param=None) -> RaySamples:
        # assert ray_samples.timesteps is not None, "Cannot warp samples if no time is given"
        assert ray_samples.frustums.offsets is None, "ray samples have already been warped"

        positions = ray_samples.frustums.get_positions()

        warped_p, warped_d = self.se3_field(positions,
                                            directions=None,
                                            warp_code=warp_code,
                                            windows_param=windows_param)

        ray_samples.frustums.set_offsets(warped_p - positions)
        # TODO: Update directions

        return ray_samples
