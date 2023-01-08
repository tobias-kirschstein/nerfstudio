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
from typing import Any, Dict, Optional, Tuple, Literal

import torch
from nerfacc import contract, ContractionType
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.hash_encoding import HashEnsembleMixingType
from nerfstudio.fields.hypernerf_field import SE3WarpingField
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP

ViewDirectionWarpType = Literal[None, 'rotation', 'samples']


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
                 aabb: torch.Tensor,
                 contraction_type: ContractionType.AABB,
                 n_freq_pos=7,
                 warp_code_dim: int = 8,
                 mlp_num_layers: int = 6,
                 mlp_layer_width: int = 128,
                 skip_connections: Tuple[int] = (4,),
                 view_direction_warping: ViewDirectionWarpType = None,
                 use_hash_encoding_ensemble: bool = False,
                 hash_encoding_ensemble_n_levels: int = 16,
                 hash_encoding_ensemble_features_per_level: int = 2,
                 hash_encoding_ensemble_n_tables: Optional[int] = None,
                 hash_encoding_ensemble_mixing_type: HashEnsembleMixingType = 'blend',
                 hash_encoding_ensemble_n_heads: Optional[int] = None,
                 only_render_hash_table: Optional[int] = None
                 ):
        super(SE3Distortion, self).__init__()

        # Parameter(..., requires_grad=False) ensures that AABB is moved to correct device
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.contraction_type = contraction_type
        self.view_direction_warping = view_direction_warping

        self.se3_field = SE3WarpingField(
            n_freq_pos=n_freq_pos,
            warp_code_dim=warp_code_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_layer_width=mlp_layer_width,
            skip_connections=skip_connections,
            warp_direction=view_direction_warping == 'rotation',
            use_hash_encoding_ensemble=use_hash_encoding_ensemble,
            hash_encoding_ensemble_n_levels=hash_encoding_ensemble_n_levels,
            hash_encoding_ensemble_features_per_level=hash_encoding_ensemble_features_per_level,
            hash_encoding_ensemble_n_tables=hash_encoding_ensemble_n_tables,
            hash_encoding_ensemble_mixing_type=hash_encoding_ensemble_mixing_type,
            hash_encoding_ensemble_n_heads=hash_encoding_ensemble_n_heads,
            only_render_hash_table=only_render_hash_table
        )

    def forward(self, ray_samples: RaySamples, warp_code=None, windows_param=None) -> RaySamples:
        # assert ray_samples.timesteps is not None, "Cannot warp samples if no time is given"
        assert ray_samples.frustums.offsets is None or (
                ray_samples.frustums.offsets == 0).all(), "ray samples have already been warped"

        positions = ray_samples.frustums.get_positions()
        # Note: contract does not propagate gradients to input positions!
        positions = contract(x=positions, roi=self.aabb, type=self.contraction_type)

        warped_p, warped_d = self.se3_field(positions,
                                            directions=ray_samples.frustums.directions,
                                            warp_code=warp_code,
                                            windows_param=windows_param)

        ray_samples.frustums.set_offsets(warped_p - positions)

        if self.view_direction_warping == 'rotation':
            warped_d = nn.functional.normalize(warped_d, dim=1, p=2)
            ray_samples.frustums.set_directions(warped_d)
        elif self.view_direction_warping == 'samples' and ray_samples.ray_indices is not None:
            idx_no_previous_sample = (ray_samples.frustums.starts - torch.roll(ray_samples.frustums.starts, 1)).squeeze(
                1).abs() > 0.011
            idx_no_subsequent_sample = (ray_samples.frustums.starts - torch.roll(ray_samples.frustums.starts,
                                                                                 -1)).squeeze(1).abs() > 0.011

            if ray_samples.ray_indices is not None:
                ray_indices = ray_samples.ray_indices.squeeze()
                idx_no_previous_sample |= ~(ray_indices == torch.roll(ray_indices, 1))
                idx_no_subsequent_sample |= ~(ray_indices == torch.roll(ray_indices, -1))

            positions = ray_samples.frustums.get_positions()
            positions_shifted_right = torch.roll(positions, 1, dims=0)
            positions_shifted_left = torch.roll(positions, -1, dims=0)
            pos_diff_previous = positions - positions_shifted_right
            pos_diff_next = positions_shifted_left - positions

            # ... a - o - o - o - b ... a - b ... ->
            # Samples which are the last samples of its group (b) will only use the direction from their previous sample
            # Respectively, samples which are the first in their group (a) will only use the direction ot their next neighbour
            # All other samples (o) will use the mean betwenn direction to previous and next neighbour
            warped_d = ray_samples.frustums.directions
            idx_only_next_sample = idx_no_previous_sample & ~idx_no_subsequent_sample
            idx_only_previous_sample = ~idx_no_previous_sample & idx_no_subsequent_sample
            idx_both_previous_and_next = ~idx_no_previous_sample & ~idx_no_subsequent_sample
            warped_d[idx_only_next_sample] = pos_diff_next[idx_only_next_sample]
            warped_d[idx_only_previous_sample] = pos_diff_previous[idx_only_previous_sample]
            warped_d[idx_both_previous_and_next] = (pos_diff_next[idx_both_previous_and_next] + pos_diff_previous[
                idx_both_previous_and_next]) / 2

            # Cloning is necessary here, otherwise we get the
            # "cannot backpropagate because one of the variables was updated with an in-place operation"
            # Somehow the division within the normalize() interfers with the index assignments before
            # Also: have to use detach() here, as there are NaNs during training otherwise ...
            warped_d = warped_d.clone().detach()
            # warped_d = warped_d / warped_d.detach().norm(dim=1, p=2).unsqueeze(1)
            warped_d = nn.functional.normalize(warped_d, p=2, dim=1)
            ray_samples.frustums.set_directions(warped_d)

        return ray_samples
