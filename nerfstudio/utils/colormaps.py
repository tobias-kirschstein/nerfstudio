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

""" Helper functions for visualizing outputs """

from typing import Optional

import numpy as np
import torch
from matplotlib import cm
from torchtyping import TensorType

from nerfstudio.utils import colors


def apply_colormap(image: TensorType["bs":..., 1], cmap="viridis") -> TensorType["bs":..., "rgb":3]:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        cmap: Colormap for image.

    Returns:
        TensorType: Colored image
    """

    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
        depth: TensorType["bs":..., 1],
        accumulation: Optional[TensorType["bs":..., 1]] = None,
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        cmap="turbo",
        acc_threshold: float = 1e-1,
) -> TensorType["bs":..., "rgb":3]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        cmap: Colormap to apply.

    Returns:
        Colored depth image
    """

    if accumulation is not None and (accumulation >= acc_threshold).sum() > 0:
        # Only use relevant part of depth to estimate min/max values for coloring.
        # Otherwise, a single outlier in a low accumulation region might spread the colormap out too much
        max_depth = depth[accumulation >= acc_threshold].max()
        min_depth = depth[accumulation >= acc_threshold].min()
    else:
        max_depth = depth.max()
        min_depth = depth.min()
    near_plane = near_plane or min_depth
    far_plane = far_plane or max_depth

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_boolean_colormap(
        image: TensorType["bs":..., 1, bool],
        true_color: TensorType["bs":..., "rgb":3] = colors.WHITE,
        false_color: TensorType["bs":..., "rgb":3] = colors.BLACK,
) -> TensorType["bs":..., "rgb":3]:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


def apply_offset_colormap(
        offsets: TensorType["bs":..., 3]
) -> TensorType["bs":..., "rgb": 3]:
    """
    Colorizes the given offsets as follows:
     - 0, 0, 0 -> white
     - 1 ,0,0 / 0,1 ,0 / 0,0, 1 -> red/green/blue
     - -1,0,0 / 0,-1,0 / 0,0,-1 -> cyan/purple/orange

    This is done by subdividing the 3D xyz offset space into 8 tetrahedrons (each contains the origin and two axis points)
    The corner points of the tetrahedrons are the above-mentioned colors and the color interpolation is found via
    barycentric coordinates of the offset point within its containing tetrahedron

    Args:
        offsets: The offsets to colorize

    Taken from: https://stackoverflow.com/questions/38545520/barycentric-coordinates-of-a-tetrahedron

    Returns:
        colors for the given offsets
    """

    points = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]], dtype=torch.float32).to(offsets)

    colors = torch.tensor([
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32).to(offsets)

    original_shape = offsets.shape
    offsets = offsets.view(-1, 3)  # Ensure we just have a list of points

    distances = (points[None, ...] - offsets[:, None, :]).norm(dim=2)  # [B, 7]
    idx_sorted = torch.sort(distances, dim=1).indices  # [B, 7]
    closest_4_points = points[idx_sorted[:, :4]]

    barycentric_coordinates = _compute_tetrahedron_barycentric_coordinates(closest_4_points[:, 0],
                                                                           closest_4_points[:, 1],
                                                                           closest_4_points[:, 2],
                                                                           closest_4_points[:, 3],
                                                                           offsets)

    barycentric_coordinates_clipped = torch.clip(barycentric_coordinates, min=0, max=1)
    barycentric_coordinates_clipped /= barycentric_coordinates_clipped.sum(dim=1).view(-1, 1)

    interpolated_colors = torch.bmm(barycentric_coordinates_clipped.view(-1, 1, 4), colors[idx_sorted[:, :4]]).squeeze(1)
    interpolated_colors = torch.clip(interpolated_colors, min=0, max=1)
    interpolated_colors = interpolated_colors.view(original_shape)

    # interpolated_colors = []
    # TODO: This naive loop is of course slow!
    # for offset in offsets:
    #     distances = (points - offset[None, ...]).norm(dim=1)
    #     idx_sorted = [idx for idx, _ in sorted(enumerate(distances), key=lambda x: x[1])]
    #     closest_4_points = points[idx_sorted[:4]]
    #
    #     barycentric_coordinates = _compute_tetrahedron_barycentric_coordinates(closest_4_points[0],
    #                                                                            closest_4_points[1],
    #                                                                            closest_4_points[2],
    #                                                                            closest_4_points[3],
    #                                                                            offset)
    #
    #     barycentric_coordinates_clipped = torch.clip(barycentric_coordinates, min=0, max=1)
    #     barycentric_coordinates_clipped /= barycentric_coordinates_clipped.sum()
    #
    #     interpolated_color = (barycentric_coordinates_clipped[..., None] * colors[idx_sorted[:4]]).sum(axis=0)
    #
    #     interpolated_colors.append(interpolated_color)
    #
    # interpolated_colors = torch.stack(interpolated_colors)
    # interpolated_colors = torch.clip(interpolated_colors, min=0, max=1)  # Ensure that colors are actually in [0, 1]
    # interpolated_colors = interpolated_colors.view(
    #     original_shape)  # Ensure that colors have the same shape as original input

    return interpolated_colors


def _scalar_triple_product(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.view(-1, 1, 3), torch.cross(b, c).view(-1, 3, 1)).squeeze(2).squeeze(1)
    # return a.dot(torch.cross(c, b))


def _compute_tetrahedron_barycentric_coordinates(a: torch.Tensor,
                                                 b: torch.Tensor,
                                                 c: torch.Tensor,
                                                 d: torch.Tensor,
                                                 p: torch.Tensor) -> torch.Tensor:
    vap = p - a
    vbp = p - b
    vab = b - a
    vac = c - a
    vad = d - a
    vbc = c - b
    vbd = d - b

    va6 = _scalar_triple_product(vbp, vbd, vbc)
    vb6 = _scalar_triple_product(vap, vac, vad)
    vc6 = _scalar_triple_product(vap, vad, vab)
    vd6 = _scalar_triple_product(vap, vab, vac)

    v6 = 1 / _scalar_triple_product(vab, vac, vad)
    return torch.stack([va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6], dim=-1).to(p)
