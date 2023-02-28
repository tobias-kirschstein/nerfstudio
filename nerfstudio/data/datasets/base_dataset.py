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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)

        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)

        if self._dataparser_outputs.color_correction_filenames is not None:
            image = np.array(pil_image, dtype="float") / 255  # shape is (h, w) or (h, w, 3 or 4)
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(3, axis=2)

            has_alpha_channels = False
            if image.shape[-1] == 4:
                alpha_channels = image[:, :, 3]
                image = image[:, :, :3]
                has_alpha_channels = True
            affine_color_transform = np.load(self._dataparser_outputs.color_correction_filenames[image_idx])
            image = image @ affine_color_transform[:3, :3] + affine_color_transform[np.newaxis, :3, 3]
            image = np.clip(image, 0, 1)
            if has_alpha_channels:
                image = np.concatenate([image, alpha_channels[:, :, np.newaxis]], axis=-1)
            image = (image*255).astype(np.uint8)
        else:
            image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(3, axis=2)

        if self._dataparser_outputs.alpha_channel_filenames is not None:
            alpha_channel_filename = self._dataparser_outputs.alpha_channel_filenames[image_idx]
            pil_alpha_image = Image.open(alpha_channel_filename)
            pil_alpha_image = pil_alpha_image.resize(pil_image.size, resample=Image.BILINEAR)

            alpha_image = np.asarray(pil_alpha_image, dtype="uint8")
            image = np.concatenate([image, alpha_image[..., None]], axis=-1)





        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """

        metadata = dict()
        image_idx = data['image_idx']
        for _, data_func_dict in self.metadata.items():
            assert "func" in data_func_dict, "Missing function to process data: specify `func` in `additional_inputs`"
            func = data_func_dict["func"]
            assert "kwargs" in data_func_dict, "No data to process: specify `kwargs` in `additional_inputs`"
            metadata.update(func(image_idx, **data_func_dict["kwargs"]))

        del data

        return metadata

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames


class InMemoryInputDataset(InputDataset):
    """
    Can be used as a drop-in replacement for InputDataset.
    Instead of loading the images everytime upon request, it caches them.
    This will increase memory consumption over time until all images have been loaded once.
    """

    def __init__(self,
                 dataparser_outputs: DataparserOutputs,
                 scale_factor: float = 1.0,
                 max_cached_items: int = -1,
                 use_cache_compression: bool = False):
        super(InMemoryInputDataset, self).__init__(dataparser_outputs, scale_factor=scale_factor)

        self._cached_items = dict()
        self._max_cached_items = max_cached_items
        self._use_cache_compression = use_cache_compression

    def _compress(self, item: Dict) -> Dict:
        if not self._use_cache_compression:
            return item

        item = item.copy()
        # Only store uint8 values for every pixel channel (discretization introduces lossy compression!)
        item['image'] = (item['image'] * 255).round().to(torch.uint8)
        return item

    def _uncompress(self, item: Dict) -> Dict:
        if not self._use_cache_compression:
            return item

        item = item.copy()
        # Cast image back into float
        item['image'] = item['image'].float() / 255.
        return item

    def __getitem__(self, image_idx):
        if image_idx in self._cached_items:
            item = self._uncompress(self._cached_items[image_idx])
        else:
            item = super().__getitem__(image_idx)
            if self._max_cached_items == -1 or len(self._cached_items) < self._max_cached_items:
                # Only cache item if number of cached items hasn't been exceeded yet
                self._cached_items[image_idx] = self._compress(item)

        return item
