import dataclasses
from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional

import torch.nn.functional as F
import tinycudann as tcnn
import torch
from nerfstudio.field_components.encodings import posenc_window
from torch import nn

HashEnsembleMixingType = Literal['blend', 'attention', 'multihead_attention']


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
                 mixing_type: HashEnsembleMixingType = 'blend',
                 dim_conditioning_code: Optional[int] = None,
                 n_heads: Optional[int] = None):
        super(HashEncodingEnsemble, self).__init__()

        self.mixing_type = mixing_type
        self.n_hash_encodings = n_hash_encodings
        self.hash_encoding_config = hash_encoding_config

        self.hash_encodings = []
        for i_hash_encoding in range(n_hash_encodings):
            self.hash_encodings.append(hash_encoding_config.setup())

        self.hash_encodings = nn.ModuleList(self.hash_encodings)

        dim_hash_encoding = hash_encoding_config.n_levels * hash_encoding_config.n_features_per_level
        if mixing_type in {'attention', 'multihead_attention'}:
            assert dim_conditioning_code is not None, "For attention mixing_type, dim_conditioning_code must be given"
            self.attention_keys = nn.Embedding(n_hash_encodings,
                                               dim_conditioning_code,
                                               #dtype=self.hash_encodings[0].dtype
                                               )
            self.sqrt_dim = sqrt(dim_conditioning_code)

            if mixing_type == 'multihead_attention':
                if n_heads is None:
                    # Assume, we just want one head per hash table level
                    n_heads = hash_encoding_config.n_levels

                self.multihead_attn = nn.MultiheadAttention(dim_conditioning_code,
                                                            n_heads,
                                                            batch_first=True,
                                                            kdim=dim_conditioning_code,
                                                            vdim=dim_hash_encoding,
                                                            # dtype=self.hash_encodings[0].dtype
                                                            )
                self.n_output_dims = dim_conditioning_code  # Multihead attention implicitly transforms the hash encodings to its internal embedding dimension

            else:
                self.n_output_dims = dim_hash_encoding
        else:
            self.n_output_dims = dim_hash_encoding

        # Unfortunately, cannot merge hashtables into a single hashtable, as the maximum number of features_per_level is 8!
        # hash_encoding_config_merged = dataclasses.replace(hash_encoding_config,
        #                                                   n_features_per_level=hash_encoding_config.n_features_per_level * n_hash_encodings)
        # self.hash_encoding = hash_encoding_config_merged.setup()

    def forward(self,
                in_tensor: torch.Tensor,
                conditioning_code: torch.Tensor,
                windows_param: Optional[float] = None) -> torch.Tensor:

        embeddings = []
        for hash_encoding in self.hash_encodings:
            embedding = hash_encoding(in_tensor)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings, dim=-1)  # [B, D, H]

        if windows_param is not None:
            window = posenc_window(windows_param,
                                   0,
                                   self.hash_encoding_config.n_levels - 1,
                                   self.hash_encoding_config.n_levels)  # [L]
            window = window.repeat_interleave(self.hash_encoding_config.n_features_per_level)  # [L*F = D]
            window = window.unsqueeze(0).unsqueeze(2).to(embeddings)  # [1, D, 1]
            embeddings = window * embeddings

        if self.mixing_type == 'blend':
            assert conditioning_code.shape[-1] == self.n_hash_encodings, \
                "If blend mixing type is chosen, conditioning code needs to have as many dimensions as there are " \
                "hashtables in the encoding"

            # embeddings = self.hash_encoding(in_tensor)  # [B, D * H]
            # n_levels = self.hash_encoding_config.n_levels
            # features_per_level = self.hash_encoding_config.n_features_per_level
            #
            # embeddings = embeddings.view(-1, n_levels, self.n_hash_encodings, features_per_level)  # [B, L, H, F]
            # embeddings = embeddings.transpose(1, 2)
            # embeddings = embeddings.reshape(-1, self.n_hash_encodings, n_levels * features_per_level)  # [B, H, L*F]
            # embeddings = embeddings.transpose(1, 2)  # [B, D, H]

            conditioning_code = conditioning_code.unsqueeze(2)  # [B, H, 1]
            conditioning_code = conditioning_code.to(embeddings)  # Make conditioning code half precision
            blended_embeddings = torch.bmm(embeddings, conditioning_code)  # [B, D, 1]
            blended_embeddings = blended_embeddings.squeeze(2)  # [B, D]

        elif self.mixing_type == 'attention':
            # Scaled dot-product attention
            keys = self.attention_keys.weight  # [H, T]
            queries = conditioning_code  # [B, T]
            values = embeddings  # [B, D, H]
            B = queries.shape[0]

            queries = queries.unsqueeze(1)  # [B, 1, T]
            values = values.to(queries)  # Make values normal precision
            keys = keys.unsqueeze(0).transpose(1, 2)  # [1, T, H]

            scores = torch.matmul(queries, keys) / self.sqrt_dim  # [B, 1, H]
            attentions = F.softmax(scores, dim=2)  # [B, 1, H]

            values = values.transpose(1, 2)  # [B, H, D]
            # TODO: This bmm might not work
            attended_embeddings = torch.bmm(attentions, values)  # [B, 1, D]
            blended_embeddings = attended_embeddings.squeeze(1)  # [B, D]
        elif self.mixing_type == 'multihead_attention':
            B = conditioning_code.shape[0]  # batch dim: number of samples

            queries = conditioning_code  # [B, C]
            keys = self.attention_keys.weight  # [H, C]
            values = embeddings  # [B, D, H]

            queries = queries.unsqueeze(1)  # [B, 1, C]
            keys = keys.unsqueeze(0)  # [1, H, C]
            # Unfortunately, we have to call repeat() here because multihead_attn does not do broadcasting
            # This leads to a massive GPU memory consumption
            keys = keys.repeat(B, 1, 1)  # [B, H, C]  share keys across samples
            values = values.transpose(1, 2)  # [B, H, D]
            values = values.to(queries)  # Make values normal precision

            # Make queries and keys half precision
            # queries = queries.to(values)
            # keys = keys.to(values)

            # NOTE: nn.MultiheadAttention implicitly uses a transform W_v for the values (hash encodings in our case)
            # which maps from vdim=H onto the embed_dim=C
            # Hence the output of the attention operation is not only a weighted combination of the original hash
            # encodings, but instead a weighted combination of TRANSFORMED (H->C) hash encodings
            # If we wanted to keep the original dimension (H) of the hash encodings, we could instead use
            # the return attention weights and perform the weighted combination ourselves
            blended_embeddings, _ = self.multihead_attn(queries, keys, values, need_weights=False)  # [B, 1, C]
            blended_embeddings = blended_embeddings.squeeze(1)  # [B, C]
        else:
            raise ValueError(f"Unsupported mixing type: {self.mixing_type}")

        return blended_embeddings

    def get_out_dim(self) -> int:
        return self.n_output_dims
