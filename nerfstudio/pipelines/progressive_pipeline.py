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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from typing_extensions import Literal

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class ProgressivePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ProgressivePipeline)
    """target class to instantiate"""


class ProgressivePipeline(VanillaPipeline):
    """The progressive training pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        self.datamanager.clear_train_batch(batch)

        # for key in list(batch.keys()):
        #     del batch[key]

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_train_image_metrics_and_images(self, step: int):
        self.eval()
        with torch.no_grad():
            image_idx, camera_ray_bundle, batch = self.datamanager.next_train_image(step)
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            assert "image_idx" not in metrics_dict
            metrics_dict["image_idx"] = image_idx
            assert "num_rays" not in metrics_dict
            metrics_dict["num_rays"] = len(camera_ray_bundle)

            # Put all eval images on CPU
            for key in images_dict.keys():
                images_dict[key] = images_dict[key].cpu()

            # No clearing necessary here, as batch is just a single image that comes directly from the InputDataset
            # These, we do not want to delete
            # Also, batch is not even on CUDA here

        self.train()
        return metrics_dict, images_dict
