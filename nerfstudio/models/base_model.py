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
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import tinycudann as tcnn
import torch
from torch import nn
from torch.nn import MSELoss, Parameter

from nerfstudio.cameras.frustum import Frustum
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    collider_type: Literal["AABBBox", "NearFar"] = "NearFar"
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    eval_scene_box_scale: Optional[float] = None
    """scene box that should be used for inference rendering. Should be smaller than train scene box"""
    use_background_network: bool = False
    background_color: Literal["random", "last_sample", "white", "black"] = "random"

    # Background model
    use_background_network: bool = False
    lambda_background_adjustment_regularization: float = 1
    num_layers_background: int = 3

    lambda_mask_loss: float = 0


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        camera_frustums: Optional[List[Frustum]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.num_train_data = num_train_data
        self.camera_frustums = camera_frustums
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if (
            self.config.enable_collider
            and self.config.collider_params is not None
            and "near_plane" in self.config.collider_params
            and "far_plane" in self.config.collider_params
        ):
            if self.config.collider_type == "AABBBox":
                self.collider = AABBBoxCollider(scene_box=self.scene_box)
            elif self.config.collider_type == "NearFar":
                assert self.config.collider_params is not None
                self.collider = NearFarCollider(
                    near_plane=self.config.collider_params["near_plane"],
                    far_plane=self.config.collider_params["far_plane"],
                )
            else:
                raise NotImplementedError(f"Unkown collider_type: {self.config.collider_type}")

        # TODO: Try Huber-Loss?
        self.rgb_loss = MSELoss()

        if self.config.use_background_network:
            self.mlp_background = tcnn.NetworkWithInputEncoding(
                n_input_dims=6,
                n_output_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {"n_dims_to_encode": 3, "otype": "Frequency", "n_frequencies": 12},
                        {"n_dims_to_encode": 3, "otype": "SphericalHarmonics", "degree": 6},
                    ],
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": self.config.num_layers_background,
                },
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

        param_groups = {"fields": []}
        if self.config.use_background_network:
            param_groups["fields"].extend(self.mlp_background.parameters())

        return param_groups

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue

            concat_output = torch.cat(outputs_list)
            assert concat_output.numel() % (image_width * image_height) == 0, (
                f"aggregated model output for channel {output_name} has {concat_output.numel()} elements "
                f"which cannot be reshaped into [{image_height}, {image_width}, -1]"
            )
            outputs[output_name] = concat_output.view(image_height, image_width, -1)  # type: ignore
        return outputs

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def apply_background_network(
        self,
        batch: Dict[str, torch.Tensor],
        rgb: torch.Tensor,
        accumulation: torch.Tensor,
        background_adjustments: Optional[torch.Tensor] = None,
    ):
        """
        Adds the color of the background images to the predicted rays if 'background_images' is supplied in `batch`.
        Additionally, if 'background_adjustments' is present in `outputs` the original background pixels will
        be adjusted with the given values.

        Args:
            batch:
                Should contain 'background_images' and 'local_indices'
            rgb:
                The color predictions per ray of the model
            accumulation:
                The accumulation per ray
            background_adjustments:
                The predictions of the background network

        Returns:
            The rgb predictions with background pixels + potential adjustments applied
        """

        if "background_images" in batch:
            background_images = batch["background_images"]  # [B, H, W, 3] or [H, W, 3] (eval)

            if self.training or "local_indices" in batch:
                local_indices = batch["local_indices"]  # [R, 3] with 3 -> (B, H, W)
                background_pixels = background_images[
                    local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]
                ]  # [R, 3]
            else:
                background_pixels = torch.tensor(background_images).to(self.device)  # [H, W, 3]

            if background_adjustments is not None:
                # background_pixels = self.softplus_bg(background_pixels + outputs["background_adjustments"])
                # TODO: subtract -0.5 from background_pixels to make the effort for the bg network symmetric?
                # background_pixels = torch.sigmoid(background_pixels - 0.5 + 10 * outputs["background_adjustments"] - 5)

                ba = background_adjustments.mean(dim=-1)
                alpha = 4 * ba.pow(2) - 4 * ba + 1  # alpha(ba=0|1) -> 1, alpha(ba=0.5) -> 0
                alpha = alpha.unsqueeze(-1)  # [R, 1]
                background_pixels = (1 - alpha) * background_pixels + alpha * background_adjustments

            rgb = rgb + (1 - accumulation) * background_pixels

        return rgb

    def apply_background_adjustment(
        self, ray_bundle: RayBundle, t_fars: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ):
        """
        Queries the background network with the given rays and stores the computed per-bg-pixel adjustments in the
        `outputs` dictionary.
        The BG network will be queried at the endpoint of the rays.

        Args:
            ray_bundle:
                should contain the origins and directions of rays
            t_fars:
                should contain the end points of rays as [R, 1] tensor
            outputs:
                dictionary to which the 'background_adjustments' will be added to
        """

        if self.config.use_background_network:
            # background network

            background_adjustments = self.mlp_background(
                torch.concat([ray_bundle.origins + t_fars * ray_bundle.directions, ray_bundle.directions], dim=1)
            )  # [R, 3]

            outputs["background_adjustments"] = background_adjustments

    def get_masked_rgb_loss(self, batch: Dict[str, torch.Tensor], rgb_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch:
                should contain the GT image as 'image'.
                If it also contains a 'mask', the RGB loss will be only computed in the masked region.
            rgb_pred:
                the models predictions for the sampled rays

        Returns:
            the loss of the models RGB predictions vs the GT colors
        """

        image = batch["image"].to(self.device)

        if "mask" in batch:
            # Only compute RGB loss on non-masked pixels
            pixel_indices_per_ray = batch["local_indices"]  # [R, [c, y, x]]
            masks = batch["mask"].squeeze(3)  # [B, H, W]
            mask = masks[
                pixel_indices_per_ray[:, 0],
                pixel_indices_per_ray[:, 1],
                pixel_indices_per_ray[:, 2],
            ]

            rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])
        else:
            rgb_loss = self.rgb_loss(image, rgb_pred)

        return rgb_loss

    def get_background_adjustment_loss(self, outputs: Dict[str, torch.Tensor]):
        if self.config.use_background_network and "background_adjustments" in outputs:
            background_adjustment_displacement = (outputs["background_adjustments"] - 0.5).pow(2).mean()
            background_adjustment_displacement = (
                self.config.lambda_background_adjustment_regularization * background_adjustment_displacement
            )

            if background_adjustment_displacement.isnan().any():
                print("WARNING! BACKGRUOND ADJUSTMENT REGULARIZATION IS NAN!")

            return background_adjustment_displacement
        else:
            return None

    def get_mask_loss(self, batch: Dict[str, torch.Tensor], accumulation: torch.Tensor) -> Optional[torch.Tensor]:
        # Mask loss
        mask_loss = None
        if self.config.lambda_mask_loss > 0 and "mask" in batch:
            accumulation_per_ray = accumulation.squeeze(1)  # [R]
            pixel_indices_per_ray = batch["local_indices"]  # [R, 3] with 3 = C,y,x
            masks = batch["mask"].squeeze(3)  # [C, H, W]

            mask_value_per_ray = masks[
                pixel_indices_per_ray[:, 0],
                pixel_indices_per_ray[:, 1],
                pixel_indices_per_ray[:, 2],
            ]

            mask_loss = (
                (1 - accumulation_per_ray[mask_value_per_ray]).sum() + (accumulation_per_ray[~mask_value_per_ray]).sum()
            ) / accumulation_per_ray.shape[0]
            mask_loss = self.config.lambda_mask_loss * mask_loss

        return mask_loss

    def apply_mask_and_combine_images(
        self,
        batch: Dict[str, torch.Tensor],
        rgb: torch.Tensor,
        accumulation: torch.Tensor,
        rgb_without_bg: Optional[torch.Tensor],
    ):

        image = batch["image"].to(self.device)

        if "mask" in batch:
            # Log GT image + full model prediction
            combined_rgb = torch.cat([image.clone(), rgb if rgb_without_bg is None else rgb_without_bg], dim=1)

            # Log masked GT image + masked model prediction which is what the evaluation is performed on
            mask = batch["mask"].squeeze(2)
            image[~mask] = 0
            rgb[~mask] = 0
            combined_rgb_masked = torch.cat([image, rgb], dim=1)

            # Density that is in the masked-out area will be summarized in a "floaters" metric
            # "floaters" is high when there is a lot of density in the masked-out region
            floaters = accumulation[~mask].mean()
        else:
            combined_rgb = torch.cat([image, rgb], dim=1)
            combined_rgb_masked = None
            floaters = None

        return image, combined_rgb, combined_rgb_masked, floaters
