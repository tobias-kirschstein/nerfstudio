from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.encodings import (
    Encoding,
    NeRFEncoding,
    WindowedNeRFEncoding,
)
from nerfstudio.field_components.mlp import MLP


class SE3Field(nn.Module):
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
        n_freq_warp=8,
        warp_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_warp, min_freq_exp=0.0, max_freq_exp=n_freq_warp - 1, include_input=Ture
        )
        self.mlp_warping = MLP(
            in_dim=self.position_encoding.get_out_dim() + warp_embed_dim,
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.uniform_(self.mlp_warping.layers[-1].weight, a=-1e-5, b=1e-5)  # for SE3 Field

    def forward(self, positions, warp_embed=None, alpha=None):
        if warp_embed is None:
            return None
        p = self.position_encoding(positions, alpha=alpha)
        return self.mlp_warping(torch.cat([p, warp_embed], dim=-1))


class DeformationField(nn.Module):
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
        n_freq_warp=8,
        warp_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_warp, min_freq_exp=0.0, max_freq_exp=n_freq_warp - 1, include_input=True
        )
        self.mlp_warping = MLP(
            in_dim=self.position_encoding.get_out_dim() + warp_embed_dim,
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

        nn.init.normal_(self.mlp_warping.layers[-1].weight, std=1e-4)

    def forward(self, positions, warp_embed=None, alpha=None):
        if warp_embed is None:
            return None
        p = self.position_encoding(positions, alpha=alpha)
        return self.mlp_warping(torch.cat([p, warp_embed], dim=-1))
        # return self.mlp_warping(torch.cat([p, warp_embed], dim=-1)) + positions
