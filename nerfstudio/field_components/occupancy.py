import torch
from nerfacc import OccupancyGrid, Grid, ContractionType
from nerfstudio.utils.connected_components import extract_top_k_connected_component


class FilteredOccupancyGrid(Grid):

    def __init__(self, occupancy_grid: OccupancyGrid):
        super(FilteredOccupancyGrid, self).__init__()
        self.occupancy_grid = occupancy_grid
        self.resolution = occupancy_grid.resolution
        self._binary = None

    @property
    def roi_aabb(self) -> torch.Tensor:
        return self.occupancy_grid.roi_aabb

    @property
    def binary(self) -> torch.Tensor:
        if self._binary is None:
            self.update()

        return self._binary

    @property
    def contraction_type(self) -> ContractionType:
        return self.occupancy_grid.contraction_type

    def update(self):
        resolution = self.occupancy_grid.resolution
        try:
            iter(resolution)
        except TypeError:
            # If resolution is not iterable, it probably was a single number
            resolution = [resolution, resolution, resolution]

        occupancy_grid_densities = self.occupancy_grid.occs
        occupancy_grid_densities = occupancy_grid_densities.reshape(*resolution)
        occupancy_grid_densities = occupancy_grid_densities.cpu().numpy()

        largest_connected_component = extract_top_k_connected_component(occupancy_grid_densities, sigma_erosion=5)[0]

        filtered_occupancy_grid = largest_connected_component > 0  # Make binary
        filtered_occupancy_grid = torch.tensor(filtered_occupancy_grid,
                                               device=self.occupancy_grid.device,
                                               dtype=self.occupancy_grid._binary.dtype)
        self._binary = self.occupancy_grid._binary & filtered_occupancy_grid
