from dataclasses import dataclass
from typing import Literal, Union

import torch
from torch import nn, TensorType
import tinycudann as tcnn


@dataclass
class TCNNHashEncodingConfig:
    n_dims_to_encode: int = 3  # Can be 3 or 4
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 1.4472692012786865
    interpolation: Literal['Linear', 'Nearest', 'Smoothstep'] = 'Linear'

    def setup(self) -> tcnn.Encoding:
        encoding = tcnn.Encoding(self.n_dims_to_encode, encoding_config={
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
            "interpolation": self.interpolation
        })

        return encoding


class HashEncodingEnsemble(nn.Module):

    def __init__(self,
                 n_hash_encodings: int,
                 dim_conditioning_code: int,
                 hash_encoding_config: TCNNHashEncodingConfig,
                 mixing_type: Literal['blend', 'attention'] = 'blend'):
        super(HashEncodingEnsemble, self).__init__()

        self.mixing_type = mixing_type
        self.n_hash_encodings = n_hash_encodings
        self.hash_encodings = []
        for i_hash_encoding in range(n_hash_encodings):
            self.hash_encodings.append(hash_encoding_config.setup())

        self.hash_encodings = nn.ModuleList(self.hash_encodings)
        self.n_output_dims = self.hash_encodings[0].n_output_dims

    def forward(self,
                in_tensor: torch.Tensor,
                conditioning_code: torch.Tensor) -> torch.Tensor:

        if self.mixing_type == 'blend':
            assert conditioning_code.shape[-1] == self.n_hash_encodings, \
                "If blend mixing type is chosen, conditioning code needs to have as many dimensions as there are " \
                "hashtables in the encoding"

            embeddings = []
            for hash_encoding in self.hash_encodings:
                embedding = hash_encoding(in_tensor)
                embeddings.append(embedding)

            embeddings = torch.stack(embeddings, dim=-1)  # [B, D, H]
            conditioning_code = conditioning_code.unsqueeze(2)  # [B, H, 1]
            conditioning_code = conditioning_code.to(embeddings)  # Make conditioning code half precision
            blended_embeddings = torch.bmm(embeddings, conditioning_code)  # [B, D, 1]
            blended_embeddings = blended_embeddings.squeeze(2)  # [B, D]

            return blended_embeddings
        else:
            raise ValueError(f"Unsupported mixing type: {self.mixing_type}")

    def get_out_dim(self) -> int:
        return self.n_output_dims
