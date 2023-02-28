#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from nerfacc import contract, OccupancyGrid
from nerfstudio.configs.experiment_config import ExperimentConfig  # pylint: disable=unused-import
from nerfstudio.data.datasets.base_dataset import InputDataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.colormaps import apply_depth_colormap
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn
from nerfstudio.utils.connected_components import extract_top_k_connected_component, filter_occupancy_grid

CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    render_width: int,
    render_height: int,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    camera_type: CameraType = CameraType.PERSPECTIVE,
    use_depth_culling: bool = False,
    use_occupancy_grid_filtering: bool = False,
    debug_occupancy_grid_filtering: bool = False
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        render_width: Video width to render.
        render_height: Video height to render.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        camera_type: Camera projection format type.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds
    scale_factor = pipeline.datamanager.config.dataparser.scale_factor
    cameras.scale_coordinate_system(scale_factor)
    n_timesteps = pipeline.datamanager.config.dataparser.n_timesteps

    if use_occupancy_grid_filtering:
        occupancy_grid : OccupancyGrid = pipeline.model.occupancy_grid

        if debug_occupancy_grid_filtering:
            np.save(f"occupancy_grid_densities_{Path(output_filename).stem}.npy", occupancy_grid.occs.cpu().numpy())
            print("Exiting rendering as debug_occupancy_grid_filtering was set")
            exit(0)

        filter_occupancy_grid(occupancy_grid, threshold=0.1)

        # resolution = occupancy_grid.resolution
        # try:
        #     iter(resolution)
        # except TypeError:
        #     # If resolution is not iterable, it probably was a single number
        #     resolution = [resolution, resolution, resolution]
        #
        # occupancy_grid_densities = occupancy_grid.occs
        # occupancy_grid_densities = occupancy_grid_densities.reshape(*resolution)
        # occupancy_grid_densities = occupancy_grid_densities.cpu().numpy()
        #
        # if debug_occupancy_grid_filtering:
        #     np.save(f"occupancy_grid_densities_{Path(output_filename).stem}.npy", occupancy_grid_densities)
        #     print("Exiting rendering as debug_occupancy_grid_filtering was set")
        #     exit(0)
        #
        # largest_connected_component = extract_top_k_connected_component(occupancy_grid_densities, sigma_erosion=5)[0]
        #
        # filtered_occupancy_grid = largest_connected_component > 0  # Make binary
        # filtered_occupancy_grid = torch.tensor(filtered_occupancy_grid,
        #                                        device=occupancy_grid.device,
        #                                        dtype=occupancy_grid._binary.dtype)
        # occupancy_grid._binary = occupancy_grid._binary & filtered_occupancy_grid

    if use_depth_culling:
        depth_maps = []
        accumulations = []
        rendered_images = []
        grid_resolution = 64
        depth_culling_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool).cuda()

        train_dataset = pipeline.datamanager.train_dataset
        n_cameras = pipeline.datamanager.dataparser.config.n_cameras
        train_cameras = train_dataset._dataparser_outputs.cameras[:n_cameras]
        for i_cam in range(n_cameras):
            # TODO: need to do this for all timesteps!
            camera_ray_bundle = train_cameras.generate_rays(camera_indices=i_cam, timesteps=0).to(torch.device('cuda'))
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            depth_map = outputs['depth']
            accumulation = outputs['accumulation']

            depth_maps.append(depth_map)
            accumulations.append(accumulation)
            rendered_images.append(outputs['rgb'])

            step_size = 0.1
            distance = depth_map.min()
            origins = camera_ray_bundle.origins
            directions = camera_ray_bundle.directions
            while True:
                distance += step_size
                ray_points = origins + distance * directions  # [H, W, 3]
                H = ray_points.shape[0]
                W = ray_points.shape[1]
                ray_points = ray_points.view(-1, 3)  # contract() requires flattened points
                ray_points_normalized = contract(x=ray_points,
                                                 roi=pipeline.model.scene_box.aabb.cuda(),
                                                 type=pipeline.model.config.contraction_type)  # [H, W, 3] - [0, 1]
                ray_points_normalized = ray_points_normalized.view(H, W, 3)
                # ray_points_normalized = (ray_points_normalized + 1) / 2  # [H, W, 3] - [0, 1]
                grid_indices = (ray_points_normalized * grid_resolution).long()  # [H, W, 3] - [0, grid_resolution]^3
                idx_inside_grid = (0 <= grid_indices) & (grid_indices < grid_resolution)
                idx_inside_grid = idx_inside_grid.all(dim=2)
                idx_before_depth = distance < depth_map.squeeze(2)  # [H, W]
                idx_enough_accumulation = accumulation.squeeze(2) > 0.1  # [H, W]
                valid_grid_indices = grid_indices[idx_inside_grid & idx_before_depth & idx_enough_accumulation]  # [?, 3]

                if distance > depth_map.mean() and not valid_grid_indices.any():
                    break

                xs = valid_grid_indices[:, 0]
                ys = valid_grid_indices[:, 1]
                zs = valid_grid_indices[:, 2]

                depth_culling_grid[xs, ys, zs] = True

        from famudy.env import FAMUDY_ANALYSES_PATH
        np.save(f"{FAMUDY_ANALYSES_PATH}/depth_culling/depth_culling_grid_timestep_0", depth_culling_grid.cpu().numpy())
        np.save(f"{FAMUDY_ANALYSES_PATH}/depth_culling/depth_maps_timestep_0", torch.concat(depth_maps, dim=-1).permute(2, 0, 1).cpu().numpy())
        np.save(f"{FAMUDY_ANALYSES_PATH}/depth_culling/accumulations_timestep_0", torch.concat(accumulations, dim=-1).permute(2, 0, 1).cpu().numpy())
        np.save(f"{FAMUDY_ANALYSES_PATH}/depth_culling/rgb_timestep_0", torch.concat(rendered_images, dim=-1).permute(2, 0, 1).cpu().numpy())

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    if output_format == "images":
        output_image_dir = output_filename.parent / output_filename.stem
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = (
            stack.enter_context(
                media.VideoWriter(
                    path=output_filename,
                    shape=(
                        int(render_height * rendered_resolution_scaling_factor),
                        int(render_width * rendered_resolution_scaling_factor) * len(rendered_output_names),
                    ),
                    fps=fps,
                )
            )
            if output_format == "video"
            else None
        )
        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                timestep = int(camera_idx / cameras.size * n_timesteps)
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, timesteps=timestep)
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                        sys.exit(1)
                    output_image = outputs[rendered_output_name].cpu().numpy()

                    if rendered_output_name == "depth":
                        output_image = apply_depth_colormap(torch.from_numpy(output_image)).numpy()

                    if output_image.shape[-1] == 1:
                        output_image = np.concatenate((output_image,) * 3, axis=-1)
                    render_image.append(output_image)
                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
                if output_format == "video" and writer is not None:
                    writer.add_image(render_image)

    if output_format == "video":
        if camera_type == CameraType.EQUIRECTANGULAR:
            insert_spherical_metadata_into_file(output_filename)


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16

    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Which checkpoint to load
    checkpoint_step: Optional[int] = None

    overwrite_config: Optional[dict] = None
    use_depth_culling: bool = False
    use_occupancy_grid_filtering: bool = False  # If true, the occupancy grid will be filtered to only contain the largest connected compoment (gets rid of isolated floaters)
    debug_occupancy_grid_filtering: bool = False,  # If true the occupancy grid densities will be written out to disk in the current working directory

    def main(self) -> None:
        """Main function."""

        _, pipeline, _ = eval_setup(
            self.load_config,
            test_mode="test" if self.traj == "spiral" else "inference",
            checkpoint_step=self.checkpoint_step,
            overwrite_config=self.overwrite_config,
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            # TODO(ethan): pass in the up direction of the camera
            camera_type = CameraType.PERSPECTIVE
            render_width = 952
            render_height = 736
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            if "camera_type" not in camera_path:
                camera_type = CameraType.PERSPECTIVE
            elif camera_path["camera_type"] == "fisheye":
                camera_type = CameraType.FISHEYE
            elif camera_path["camera_type"] == "equirectangular":
                camera_type = CameraType.EQUIRECTANGULAR
            else:
                camera_type = CameraType.PERSPECTIVE
            render_width = camera_path["render_width"]
            render_height = camera_path["render_height"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        # n_frames = 5
        # camera_path = Cameras(camera_path.camera_to_worlds[:n_frames],
        #                       camera_path.fx[:n_frames],
        #                       camera_path.fy[:n_frames],
        #                       camera_path.cx[:n_frames],
        #                       camera_path.cy[:n_frames],
        #                       width=camera_path.image_width[:n_frames],
        #                       height=camera_path.image_height[:n_frames],
        #                       camera_type=camera_path.camera_type[:n_frames])

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            camera_type=camera_type,
            render_width=render_width,
            render_height=render_height,
            use_depth_culling=self.use_depth_culling,
            use_occupancy_grid_filtering=self.use_occupancy_grid_filtering,
            debug_occupancy_grid_filtering=self.debug_occupancy_grid_filtering
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
