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
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.pipelines.base_pipeline import Pipeline

CONSOLE = Console(width=120)


def eval_load_checkpoint(config: cfg.TrainerConfig, pipeline: Pipeline) -> Path:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    view_frustum_culling: Optional[bool] = None,
    checkpoint_step: Optional[int] = None,
    overwrite_collider_type: Literal["NearFar", "AABBBox", None] = None,
    eval_scene_box_scale: Optional[float] = None,
    near_plane: Optional[float] = None,
    density_threshold: Optional[float] = None,
) -> Tuple[cfg.Config, Pipeline, Path]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory

        checkpoint_step: Which checkpoint to load. Default is the latest
        eval_scene_box_scale:
            None: keep it the way it is defined in config.yaml
            -1: remove eval_scene_box (i.e., it will be the same as the regular scene box)
            any number: size of the eval scene box (should be smaller than regular scene box)

    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if view_frustum_culling is not None:
        config.pipeline.datamanager.dataparser.view_frustum_culling = view_frustum_culling

    if overwrite_collider_type is not None:
        config.pipeline.model.collider_type = overwrite_collider_type

    if eval_scene_box_scale is not None:
        if eval_scene_box_scale < 0:
            config.pipeline.model.eval_scene_box_scale = None
        else:
            config.pipeline.model.eval_scene_box_scale = eval_scene_box_scale

    if near_plane is not None:
        config.pipeline.model.near_plane = near_plane

    if density_threshold is not None:
        config.pipeline.model.density_threshold = density_threshold

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.trainer.load_dir = config.get_checkpoint_dir()
    config.trainer.load_step = checkpoint_step
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path = eval_load_checkpoint(config.trainer, pipeline)

    return config, pipeline, checkpoint_path
