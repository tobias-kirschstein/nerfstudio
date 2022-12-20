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

"""
Code for sampling pixels.
"""

import random
from typing import Dict

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

debug_sample_probabilities = None

def collate_image_dataset_batch(batch: Dict,
                                num_rays_per_batch: int,
                                keep_full_image: bool = False,
                                sample_masked_pixels: bool = True):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch

    if "pixel_sample_probabilities" in batch:
        pixel_sample_probabilities = batch["pixel_sample_probabilities"]  # [C, H, W]

        if 'mask' in batch and not sample_masked_pixels:
            mask = batch['mask']
            pixel_sample_probabilities[~mask.squeeze(-1)] = 0  # Do not sample masked out areas

        downscale_factor = 4
        resizer = Resize((image_height // downscale_factor, image_width // downscale_factor),
                         interpolation=InterpolationMode.NEAREST)
        pixel_sample_probabilities = resizer(pixel_sample_probabilities)

        B, H, W = pixel_sample_probabilities.shape

        y_offsets = torch.randint(downscale_factor, (H,))
        x_offsets = torch.randint(downscale_factor, (W,))

        # ys = torch.arange(H) * downscale_factor + y_offsets
        # xs = torch.arange(W) * downscale_factor + x_offsets
        # pixel_sample_probabilities = pixel_sample_probabilities[:, ys, xs].reshape(B, H, W)

        grid_b, grid_y, grid_x = torch.meshgrid(torch.arange(B),
                                                torch.arange(H) * downscale_factor + y_offsets,
                                                torch.arange(W) * downscale_factor + x_offsets)

        pixel_sample_indices = torch.multinomial(pixel_sample_probabilities.view(-1),
                                                 num_rays_per_batch,
                                                 replacement=True)
        grid = torch.stack([grid_b, grid_y, grid_x], dim=-1)
        indices = grid.view(-1, 3)[pixel_sample_indices]
    elif "mask" in batch and not sample_masked_pixels:
        nonzero_indices = torch.nonzero(batch["mask"][..., 0], as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

    # global debug_sample_probabilities
    # B, H, W, _ = batch['image'].shape
    # if debug_sample_probabilities is None:
    #     debug_sample_probabilities = np.zeros((B, H, W))
    #
    # debug_sample_probabilities[batch['image_idx'][indices[:, 0]].cpu(), indices[:, 1].cpu(), indices[:, 2].cpu()] += 1

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key not in {'image_idx', 'cam_ids', 'timesteps'} and value is not None}

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if 'cam_ids' in batch:
        collated_batch["cam_ids"] = batch['cam_ids'][c]
    if 'timesteps' in batch:
        collated_batch['timesteps'] = batch['timesteps'][c]

    # Needed to correct the random indices to their actual camera idx locations.
    local_indices = indices.clone()
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices
    collated_batch["local_indices"] = local_indices

    # v0.1.9 change
    # collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}
    #
    # assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape
    #
    # # Needed to correct the random indices to their actual camera idx locations.
    # indices[:, 0] = batch["image_idx"][c]
    # collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


def collate_image_dataset_batch_list(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    device = batch["image"][0].device
    num_images = len(batch["image"])

    # only sample within the mask, if the mask is in the batch
    all_indices = []
    all_images = []

    if "mask" in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            nonzero_indices = torch.nonzero(batch["mask"][i][..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch["image"][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(
                torch.rand((num_rays_in_batch, 3), device=device)
                * torch.tensor([1, image_height, image_width], device=device)
            ).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key != "image_idx" and key != "image" and key != "mask" and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int,
                 keep_full_image: bool = False,
                 sample_masked_pixels: bool = True) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image
        self.sample_masked_pixels = sample_masked_pixels

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = collate_image_dataset_batch(
                image_batch,
                self.num_rays_per_batch,
                keep_full_image=self.keep_full_image,
                sample_masked_pixels=self.sample_masked_pixels
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch
