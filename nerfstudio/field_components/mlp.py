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
Multi Layer Perceptron
"""
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent
import tinycudann as tcnn


class MLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Ouput layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
            self,
            in_dim: int,
            num_layers: int,
            layer_width: int,
            out_dim: Optional[int] = None,
            skip_connections: Optional[Tuple[int]] = None,
            activation: Optional[nn.Module] = nn.ReLU(),
            out_activation: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


@dataclass
class TCNNMLPConfig:
    n_input_dims: int
    n_output_dims: int
    n_layers: int  # Number of weight matrices
    layer_width: int
    skip_connections: Optional[Tuple[int]] = None
    activation: Optional[str] = 'ReLU'
    out_activation: Optional[str] = None


class TCNNMLP(FieldComponent):

    def __init__(self,
                 config: TCNNMLPConfig
                 ):
        super(TCNNMLP, self).__init__(in_dim=config.n_input_dims, out_dim=config.n_output_dims)
        self._config = config
        self._mlps = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        base_network_config = {
            "otype": "FullyFusedMLP" if self._config.layer_width <= 128 else "CutlassMLP",
            "activation": self._config.activation,
            "output_activation": self._config.out_activation,
            "n_neurons": self._config.layer_width,
            # "n_hidden_layers": self._config.n_layers - 1
        }
        self._mlps = []
        if self._config.skip_connections is None:
            skip_connections = []
        else:
            skip_connections = self._config.skip_connections

        previous_mlp_out_dim = 0
        previous_skip_connection = 0
        for skip_connection in skip_connections:
            network_config = base_network_config.copy()
            network_config["n_hidden_layers"] = skip_connection - previous_skip_connection - 1
            mlp = tcnn.Network(previous_mlp_out_dim + self._config.n_input_dims,
                               self._config.layer_width,
                               network_config)

            previous_mlp_out_dim = self._config.layer_width
            previous_skip_connection = skip_connection
            self._mlps.append(mlp)

        network_config = base_network_config.copy()
        network_config["n_hidden_layers"] = self._config.n_layers - previous_skip_connection - 1
        mlp = tcnn.Network(previous_mlp_out_dim + self._config.n_input_dims,
                           self._config.n_output_dims,
                           network_config)
        self._mlps.append(mlp)
        # else:
        #     network_config = base_network_config.copy()
        #     network_config["n_hidden_layers"] = self._config.n_layers - 1
        #     mlp = tcnn.Network(self._config.n_input_dims, self._config.n_output_dims, network_config)
        #     self._mlps.append(mlp)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        assert self._mlps is not None, "build_nn_modules() not called"

        original_shape = in_tensor.shape
        in_tensor = in_tensor.view(-1, original_shape[-1])  # Flatten everything into batch dimension
        # x = in_tensor
        x = torch.zeros((in_tensor.shape[0], 0), device=in_tensor.device, dtype=in_tensor.dtype)
        for i, mlp in enumerate(self._mlps):
            x = torch.cat([in_tensor, x], -1)  # Add original input for skip connection
            x = mlp(x)  # Forward

        out_shape = list(original_shape)
        out_shape[-1] = self._config.n_output_dims
        x = x.view(out_shape)

        return x
