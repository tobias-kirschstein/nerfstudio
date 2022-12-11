from typing import Any, Dict, Optional, Tuple

import pytorch3d
import pytorch3d.transforms
import torch
from torch import nn

from nerfstudio.field_components.encodings import WindowedNeRFEncoding
from nerfstudio.field_components.mlp import MLP


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]


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
        time_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_warp, min_freq_exp=0.0, max_freq_exp=n_freq_warp - 1, include_input=True
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

        return warped_p.reshape(*positions.shape[:2], 3)


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
        time_embed_dim: int = 8,
        mlp_num_layers: int = 6,
        mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_warp, min_freq_exp=0.0, max_freq_exp=n_freq_warp - 1, include_input=True
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
