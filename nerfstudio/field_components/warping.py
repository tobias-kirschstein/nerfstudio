from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP


class NeRFieWarping(nn.Module):
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
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=False
        ),
        warp_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.mlp_warping = MLP(
            in_dim=self.position_encoding.get_out_dim() + warp_embed_dim,
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def forward(self, positions, warp_embed=None):
        if warp_embed is None:
            return None
        p = self.position_encoding(positions)
        return self.mlp_deform(torch.cat([p, warp_embed], dim=-1))