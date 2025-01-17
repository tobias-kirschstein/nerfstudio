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
from elias.config import implicit
from torch import nn
from torch.nn import MSELoss, Parameter

from nerfstudio.cameras.frustum import Frustum
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider


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
    use_backgrounds: bool = implicit(False)
    """Information from dataparser whether backgrounds are available. Only then the bg network can be used"""
    lambda_background_adjustment_regularization: float = 1
    num_layers_background: int = 3

    lambda_mask_loss: float = 0
    lambda_alpha_loss: Optional[float] = None
    use_l1_for_alpha_loss: bool = False
    mask_rgb_loss: bool = False  # Whether to only compute the RGB loss on foreground pixels if a mask is provided
    enforce_non_masked_density: bool = False
    """Whether the mask loss should enforce density in non-masked regions to be high"""

    lambda_beta_loss: float = 0
    """Enforces density to be either large (opaque) or small (transparent). Discourages semi-transparent floaters"""

    lambda_temporal_tv_loss: float = 0  # Enforce total variation loss across temporal codes


    n_parameters: int = (
        implicit()
    )  # Total number of trainable parameters of the model. Is filled in by the pipeline automatically and logged to wandb


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
        self.render_aabb = None  # the box that we want to render - should be a subset of scene_box
        self.num_train_data = num_train_data
        self.camera_frustums = camera_frustums
        self.kwargs = kwargs
        self.collider = None

        self.callbacks = []
        self.populate_modules()  # populate the modules
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

        if self.config.enable_collider:

            if self.config.collider_type == "AABBBox":
                if self.config.collider_params is not None and "near_plane" in self.config.collider_params:
                    near_plane = self.config.collider_params["near_plane"]
                else:
                    near_plane = 0

                self.collider = AABBBoxCollider(scene_box=self.scene_box, near_plane=near_plane)

            elif self.config.collider_type == "NearFar":
                assert self.config.collider_params is not None
                assert "near_plane" in self.config.collider_params
                assert "far_plane" in self.config.collider_params

                self.collider = NearFarCollider(
                    near_plane=self.config.collider_params["near_plane"],
                    far_plane=self.config.collider_params["far_plane"],
                )

            else:
                raise NotImplementedError(f"Unkown collider_type: {self.config.collider_type}")

        # TODO: Try Huber-Loss?
        self.rgb_loss = MSELoss()

        if self.config.use_backgrounds and self.config.use_background_network:
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

        param_groups = defaultdict(list)

        if self.config.use_backgrounds and self.config.use_background_network:
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

        # To avoid out of memory during evaluation, we randomly sample rays
        # Compared to going "line by line" this avoids situations where all rays are completely dense
        # Instead, it mirrors the way rays are rendered during training, i.e., if training doesn't give OOM
        # then this kind of ray sampling during evluation also won't cause OOM
        shuffled_rays = camera_ray_bundle.flatten()
        shuffled_ray_indices = torch.randperm(len(shuffled_rays))
        shuffled_rays = shuffled_rays[shuffled_ray_indices]

        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = shuffled_rays[start_idx: end_idx]
            # ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            torch.cuda.empty_cache()
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue

            concat_output = torch.cat(outputs_list)
            # Undo the shuffling in order to get the correct image
            rearranged_output = torch.zeros_like(concat_output)
            rearranged_output[shuffled_ray_indices] = concat_output
            concat_output = rearranged_output

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

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """

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

        if self.config.use_backgrounds and self.config.use_background_network:
            # background network

            background_adjustments = self.mlp_background(
                torch.concat([ray_bundle.origins + t_fars * ray_bundle.directions, ray_bundle.directions], dim=1)
            )  # [R, 3]

            outputs["background_adjustments"] = background_adjustments

    def get_mask_per_ray(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if "mask" in batch:
            pixel_indices_per_ray = batch["local_indices"]  # [R, [c, y, x]]
            masks = batch["mask"].squeeze(3)  # [B, H, W]
            mask = masks[
                pixel_indices_per_ray[:, 0],
                pixel_indices_per_ray[:, 1],
                pixel_indices_per_ray[:, 2],
            ]

            return mask
        else:
            return None

    def get_alpha_per_ray(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        assert "alpha_map" in batch
        pixel_indices_per_ray = batch["local_indices"]  # [R, [c, y, x]]
        alpha_maps = batch["alpha_map"].squeeze(3)  # [B, H, W]
        a = (
                alpha_maps[
                    pixel_indices_per_ray[:, 0],
                    pixel_indices_per_ray[:, 1],
                    pixel_indices_per_ray[:, 2],
                ].float()
                / 255
        )

        return a

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

        if 'loss_weight_map' in batch:
            pixel_indices_per_ray = batch["local_indices"]  # [R, [c, y, x]]
            loss_weight_maps = batch["loss_weight_map"]  # [B, H, W]
            loss_weight_map = loss_weight_maps[
                pixel_indices_per_ray[:, 0],
                pixel_indices_per_ray[:, 1],
                pixel_indices_per_ray[:, 2],
            ]

            loss_weight_map = loss_weight_map.unsqueeze(1)  # [R, 1]
            loss_weight_map = loss_weight_map.sqrt()  # Influence of weight map will be squared in MSE. Hence sqrt here

            # Scale both predicted and target values with the lambda values from the loss_weight_map
            # This should lead to larger updates for these rays
            image = loss_weight_map * image
            rgb_pred = loss_weight_map * rgb_pred

        # TODO (22.01.2023): Use alpha mask for computing masked RGB loss?
        if self.config.mask_rgb_loss and "mask" in batch:
            # Only compute RGB loss on non-masked pixels
            mask = self.get_mask_per_ray(batch)

            rgb_loss = self.rgb_loss(image[mask], rgb_pred[mask])
        else:
            rgb_loss = self.rgb_loss(image, rgb_pred)

        return rgb_loss

    def get_background_adjustment_loss(self, outputs: Dict[str, torch.Tensor]):
        if (
                self.config.use_background_network
                and "background_adjustments" in outputs
                and self.config.lambda_background_adjustment_regularization > 0
        ):
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
            mask_value_per_ray = self.get_mask_per_ray(batch)

            # Accumulation in masked regions should be low
            mask_loss = accumulation_per_ray[~mask_value_per_ray].sum()

            if self.config.enforce_non_masked_density:
                # Accumulation in non-masked regions should be high
                mask_loss += (1 - accumulation_per_ray[mask_value_per_ray]).sum()
                if accumulation_per_ray.shape[0] > 0:
                    mask_loss /= accumulation_per_ray.shape[0]  # Compute mask loss per ray
            else:
                n_masked_rays = (~mask_value_per_ray).sum()
                if n_masked_rays > 0:
                    mask_loss /= n_masked_rays  # Compute mask loss per ray

            mask_loss = self.config.lambda_mask_loss * mask_loss

            if mask_loss.isnan():
                print(
                    f"WARNING! MASK LOSS IS NAN! accumulation_per_ray: {accumulation_per_ray}, mask_value_per_ray: {mask_value_per_ray}"
                )
                mask_loss = 0

        return mask_loss

    def get_alpha_loss(self, batch: Dict[str, torch.Tensor], accumulation: torch.Tensor) -> Optional[torch.Tensor]:
        alpha_loss = None

        if self.config.lambda_alpha_loss is not None and self.config.lambda_alpha_loss > 0:

            accumulation_per_ray = accumulation.squeeze(1)  # [R]
            alpha_per_ray = self.get_alpha_per_ray(batch)
            if self.config.enforce_non_masked_density:
                # Compute alpha loss everywhere
                if self.config.use_l1_for_alpha_loss:
                    alpha_loss = (accumulation_per_ray - alpha_per_ray).abs().mean() * self.config.lambda_alpha_loss
                else:
                    alpha_loss = ((accumulation_per_ray - alpha_per_ray) ** 2).mean() * self.config.lambda_alpha_loss
            else:
                # Only compute alpha loss in areas where the accumulation should be below 1
                idx_background = alpha_per_ray < 1

                if 'mask' in batch and self.config.lambda_mask_loss > 0:
                    # If both mask and alpha_mask are used, don't enforce density in regions where mask says it should
                    # be empty
                    mask_per_ray = self.get_mask_per_ray(batch)
                    if (~mask_per_ray & ~idx_background).any():
                        print("[WARNING] There were rays where alpha map says foreground, but mask says background, which shouldn't happen because the background mask is a subset of the background alpha mask")
                    idx_background &= mask_per_ray
                    alpha_per_ray = (alpha_per_ray - 0.5) * 2 # Scale alpha map such that there is a smooth transition, when cutoff was at 128

                if idx_background.any():
                    if self.config.use_l1_for_alpha_loss:
                        alpha_loss = (accumulation_per_ray[idx_background] - alpha_per_ray[
                            idx_background]).abs().mean() * self.config.lambda_alpha_loss
                    else:
                        alpha_loss = ((accumulation_per_ray[idx_background] - alpha_per_ray[
                            idx_background]) ** 2).mean() * self.config.lambda_alpha_loss

        return alpha_loss

    def get_floaters_metric(self, batch: Dict[str, torch.Tensor], accumulation: torch.Tensor) -> Optional[torch.Tensor]:
        mask = self.get_mask_per_ray(batch)
        if mask is not None:
            floaters = accumulation[~mask].mean()

            return floaters
        else:
            return None

    def get_beta_loss(self, accumulation: torch.Tensor) -> Optional[torch.Tensor]:
        beta_loss = None
        if self.config.lambda_beta_loss > 0 and self.training:
            accumulation_per_ray = accumulation.squeeze(1)  # [R]
            beta_loss = ((0.1 + accumulation_per_ray).log() + (1.1 - accumulation_per_ray).log() + 2.20727).mean()
            beta_loss = self.config.lambda_beta_loss * beta_loss

        return beta_loss

    def get_landmark_loss(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.sched_window_deform is not None:
            # TODO: Maybe go back to using get_value() which outputs final_value for evaluation
            # window_deform = self.sched_window_deform.get_value() if self.sched_window_deform is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
        else:
            window_deform = None

        landmarks_a = batch["landmarks"]  # B x N_lm x 3
        rnd_perm = torch.randperm(landmarks_a.shape[0], device=landmarks_a.device)
        landmarks_b = batch["landmarks"][rnd_perm]
        time_embeddings_a = (
            self.time_embedding(batch["timesteps"]).unsqueeze(1).repeat(1, landmarks_a.shape[1], 1)
        )  # B x N_LM  x embed_dim
        time_embeddings_b = (
            self.time_embedding(batch["timesteps"][rnd_perm]).unsqueeze(1).repeat(1, landmarks_b.shape[1], 1)
        )

        valid_a = ~(landmarks_a.isnan().any(dim=-1))
        valid_b = ~(landmarks_b.isnan().any(dim=-1))
        landmarks_a_clone = landmarks_a.detach().clone()
        landmarks_b_clone = landmarks_b.detach().clone()

        landmarks_a = landmarks_a[valid_a, :].view(-1, 3)
        landmarks_b = landmarks_b[valid_b, :].view(-1, 3)

        time_embeddings_a = time_embeddings_a[valid_a, :].view(landmarks_a.shape[0], -1)
        time_embeddings_b = time_embeddings_b[valid_b, :].view(landmarks_b.shape[0], -1)

        # landmarks_a = contract(x=landmarks_a,
        #                          roi=self.temporal_distortion.aabb,
        #                          type=self.temporal_distortion.contraction_type)

        # landmarks_b = contract(x=landmarks_b,
        #                          roi=self.temporal_distortion.aabb,
        #                          type=self.temporal_distortion.contraction_type)

        landmarks_a = (landmarks_a - self.temporal_distortion.aabb[0]) / (
                self.temporal_distortion.aabb[1] - self.temporal_distortion.aabb[0]
        )
        landmarks_b = (landmarks_b - self.temporal_distortion.aabb[0]) / (
                self.temporal_distortion.aabb[1] - self.temporal_distortion.aabb[0]
        )

        warped_landmarks_a, _ = self.temporal_distortion.se3_field(
            landmarks_a, directions=None, warp_code=time_embeddings_a, windows_param=window_deform
        )

        warped_landmarks_b, _ = self.temporal_distortion.se3_field(
            landmarks_b, directions=None, warp_code=time_embeddings_b, windows_param=window_deform
        )

        valid_ab = ~(landmarks_a_clone[valid_b, :].isnan().any(dim=-1))
        valid_ba = ~(landmarks_b_clone[valid_a, :].isnan().any(dim=-1))

        warped_landmarks_a = warped_landmarks_a[valid_ba]
        warped_landmarks_b = warped_landmarks_b[valid_ab]

        loss = ((warped_landmarks_a - warped_landmarks_b)).abs()
        assert not loss.isnan().any()
        return loss

    def get_landmark_loss_direct(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.sched_window_deform is not None:
            # TODO: Maybe go back to using get_value() which outputs final_value for evaluation
            # window_deform = self.sched_window_deform.get_value() if self.sched_window_deform is not None else None
            window_deform = self.sched_window_deform.value if self.sched_window_deform is not None else None
        else:
            window_deform = None

        canonical_timestep_mask = batch["timesteps"] == 1

        landmarks_src = batch["landmarks"][~canonical_timestep_mask, :, :]  # T_non_can x N_lm x 3
        landmarks_tgt = batch["landmarks"][canonical_timestep_mask, :, :]  # T_can x N_lm x 3
        landmarks_tgt = landmarks_tgt[0:1, ...].repeat(
            landmarks_src.shape[0], 1, 1
        )  # there is only one canoncial timestep

        landmarks = torch.stack([landmarks_tgt, landmarks_src], dim=0)  # 2 x T_non_can x N_lm x 3

        time_embeddings = (
            self.time_embedding(batch["timesteps"][~canonical_timestep_mask])
                .unsqueeze(1)
                .repeat(1, landmarks_src.shape[1], 1)
        )  # T_non_can x N_lm x embed_dim

        valid = ~torch.isnan(landmarks).any(-1).any(0)  # T_non_can x N_lm

        landmarks_src = landmarks[1][valid, :]  # N_valid_lms x 3
        landmarks_tgt = landmarks[0][valid, :]  # N_valid_lms x 3

        time_embeddings = time_embeddings[valid, :]  # N_valid_lms x 3

        # landmarks_src = contract(x=landmarks_src,
        #                          roi=self.temporal_distortion.aabb,
        #                          type=self.temporal_distortion.contraction_type)
        # landmarks_tgt = contract(x=landmarks_tgt,
        #                         roi=self.temporal_distortion.aabb,
        #                         type=self.temporal_distortion.contraction_type)
        landmarks_src = (landmarks_src - self.temporal_distortion.aabb[0]) / (
                self.temporal_distortion.aabb[1] - self.temporal_distortion.aabb[0]
        )
        landmarks_tgt = (landmarks_tgt - self.temporal_distortion.aabb[0]) / (
                self.temporal_distortion.aabb[1] - self.temporal_distortion.aabb[0]
        )

        warped_landmarks_src, _ = self.temporal_distortion.se3_field(
            landmarks_src, directions=None, warp_code=time_embeddings, windows_param=window_deform
        )

        loss = ((warped_landmarks_src - landmarks_tgt)).abs()
        assert not loss.isnan().any()
        return loss

    def get_temporal_tv_loss(self, embedding: nn.Embedding, use_sparsity_prior=True):

        if self.config.lambda_temporal_tv_loss > 0:

            timesteps1 = embedding(
                torch.arange(embedding.num_embeddings - 1, device=embedding.weight.device)
            )
            timesteps2 = embedding(
                torch.arange(1, embedding.num_embeddings, device=embedding.weight.device)
            )

            temporal_difference = (timesteps1 - timesteps2).norm(dim=-1)
            l1_sparsity = timesteps1.abs().sum(dim=-1)

            temporal_tv_loss = self.config.lambda_temporal_tv_loss * temporal_difference.mean()

            if use_sparsity_prior:
                temporal_tv_loss += (self.config.lambda_temporal_tv_loss / 10) * l1_sparsity.mean()

            return temporal_tv_loss
        else:
            return None

        # timesteps1 = self.time_embedding(
        #     torch.arange(self.time_embedding.num_embeddings - 1, device=self.time_embedding.weight.device)
        # )
        # timesteps2 = self.time_embedding(
        #     torch.arange(1, self.time_embedding.num_embeddings, device=self.time_embedding.weight.device)
        # )
        #
        # temporal_difference = (timesteps1 - timesteps2).square().sum(dim=-1).sqrt()
        # if return_sparsity_prior:
        #     return temporal_difference, timesteps1.abs().sum(dim=-1)
        # else:
        #     return temporal_difference

    def apply_mask(
            self, batch: Dict[str, torch.Tensor], rgb: torch.Tensor, accumulation: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if "alpha_map" in batch:
            alpha_mask = batch["alpha_map"] / 255.  # [H, W, 1]
            alpha_mask = torch.from_numpy(alpha_mask).to(rgb)

            image_masked = batch["image"].clone().to(self.device)
            rgb_masked = rgb.clone()

            image_masked = alpha_mask * image_masked + (1 - alpha_mask)
            rgb_masked = alpha_mask * rgb_masked + (1 - alpha_mask)

            mask = alpha_mask.squeeze(2) > 0.5
            floaters = accumulation[~mask].mean()

            return image_masked, rgb_masked, floaters

        elif "mask" in batch:
            # Log masked GT image + masked model prediction which is what the evaluation is performed on
            mask = batch["mask"].squeeze(2)

            image_masked = batch["image"].clone().to(self.device)
            rgb_masked = rgb.clone()

            image_masked[~mask] = 1
            rgb_masked[~mask] = 1

            # Density that is in the masked-out area will be summarized in a "floaters" metric
            # "floaters" is high when there is a lot of density in the masked-out region
            floaters = accumulation[~mask].mean()

            return image_masked, rgb_masked, floaters

        return None, None, None

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
