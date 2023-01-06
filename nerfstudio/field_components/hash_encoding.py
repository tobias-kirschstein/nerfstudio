from dataclasses import dataclass
from typing import Literal, Union, Optional

import torch
from nerfstudio.field_components.encodings import posenc_window
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
                 hash_encoding_config: TCNNHashEncodingConfig,
                 mixing_type: Literal['blend', 'attention'] = 'blend'):
        super(HashEncodingEnsemble, self).__init__()

        self.mixing_type = mixing_type
        self.n_hash_encodings = n_hash_encodings
        self.hash_encodings = []
        self.hash_encoding_config = hash_encoding_config
        for i_hash_encoding in range(n_hash_encodings):
            self.hash_encodings.append(hash_encoding_config.setup())

        self.hash_encodings = nn.ModuleList(self.hash_encodings)
        self.n_output_dims = self.hash_encodings[0].n_output_dims

    def forward(self,
                in_tensor: torch.Tensor,
                conditioning_code: torch.Tensor,
                windows_param: Optional[float] = None) -> torch.Tensor:

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

            if windows_param is not None:
                window = posenc_window(windows_param,
                                       0,
                                       self.hash_encoding_config.n_levels - 1,
                                       self.hash_encoding_config.n_levels)
                window = window.repeat_interleave(self.hash_encoding_config.n_features_per_level)
                window = window.unsqueeze(0).to(blended_embeddings)
                blended_embeddings = window * blended_embeddings

            return blended_embeddings
        else:
            raise ValueError(f"Unsupported mixing type: {self.mixing_type}")

    def get_out_dim(self) -> int:
        return self.n_output_dims


class MLPWithHashEncodingEnsemble(nn.Module):

    def __init__(self,
                 mlp: tcnn.NetworkWithInputEncoding,
                 hash_encoding_ensemble: HashEncodingEnsemble):
        super(MLPWithHashEncodingEnsemble, self).__init__()
        self.mlp = mlp
        self.hash_encoding_ensemble = hash_encoding_ensemble

    def forward(self, in_tensor: torch.Tensor):
        pass
