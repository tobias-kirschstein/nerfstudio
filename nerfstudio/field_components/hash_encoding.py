import dataclasses
from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional, Tuple

import torch.nn.functional as F
import tinycudann as tcnn
import torch
from nerfstudio.field_components.encodings import posenc_window
from torch import nn
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import NeRFEncoding

HashEnsembleMixingType = Literal['blend', 'attention', 'multihead_attention',
                                    'multihead_blend',
                                    'multihead_blend_mixed',
                                    'multihead_blend_attention_style',
                                    'mlp_blend_field'
]


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)


class IdenEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 3

    def forward(self, x):
        return x

    def get_out_dim(self):
        return self.out_dim


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


@dataclass
class BlendFieldConfig:
    n_hidden_dims: int = 64
    n_layers: int = 4
    output_activation: Optional[Literal['Normalization', 'Tanh']] = 'Tanh'
    n_freq_pos_enc: int = 0
    skip_connections: Optional[Tuple[int]] = None
    input_dim: int = 3

    def setup(self, n_time_condition_dim: int, n_hash_tables: int) -> (NeRFEncoding, MLP):
        if self.n_freq_pos_enc > 0:
            nerf_encoder = NeRFEncoding(in_dim=self.input_dim,
                                        num_frequencies=self.n_freq_pos_enc,
                                        min_freq_exp=0,
                                        max_freq_exp=self.n_freq_pos_enc - 1,
                                        include_input=True)
        else:
            nerf_encoder = IdenEnc()
        if self.output_activation is None:
            out_activation = None
        elif self.output_activation == 'Normalization':
            out_activation = Normalizer()
        elif self.output_activation == 'Tanh':
            out_activation = nn.Tanh()
        mlp_blend_field = MLP(
            in_dim=n_time_condition_dim + nerf_encoder.get_out_dim() if (self.n_freq_pos_enc > 0) else n_time_condition_dim + self.input_dim,
            num_layers=self.n_layers,
            layer_width=self.n_hidden_dims,
            out_dim=n_hash_tables,
            skip_connections=self.skip_connections,
            out_activation=out_activation
        )
        return nerf_encoder, mlp_blend_field


class HashEncodingEnsemble(nn.Module):

    def __init__(self,
                 n_hash_encodings: int,
                 hash_encoding_config: TCNNHashEncodingConfig,
                 mixing_type: HashEnsembleMixingType = 'blend',
                 dim_conditioning_code: Optional[int] = None,
                 n_heads: Optional[int] = None,
                 only_render_hash_table: Optional[int] = None,
                 blend_field_config: BlendFieldConfig = None):
        super(HashEncodingEnsemble, self).__init__()

        self.mixing_type = mixing_type
        self.n_hash_encodings = n_hash_encodings
        self.hash_encoding_config = hash_encoding_config
        self.only_render_hash_table = only_render_hash_table

        self.hash_encodings = []
        for i_hash_encoding in range(n_hash_encodings):
            self.hash_encodings.append(hash_encoding_config.setup())

        self.hash_encodings = nn.ModuleList(self.hash_encodings)

        dim_hash_encoding = hash_encoding_config.n_levels * hash_encoding_config.n_features_per_level
        if mixing_type in ['multihead_blend', 'multihead_blend_mixed']:
            if n_heads is None:
                # Assume, we just want one head per hash table level
                n_heads = hash_encoding_config.n_levels

            self.n_heads = n_heads
            assert dim_hash_encoding % self.n_heads == 0, \
                "output hash encoding dimensionality must be divisible by n_heads"
            self.n_features_per_head = int(dim_hash_encoding / self.n_heads)
            self.n_output_dims = dim_hash_encoding

            if mixing_type == 'multihead_blend_mixed':
                self.mixing_heads = nn.ModuleList(
                    [nn.Linear(dim_hash_encoding, dim_hash_encoding) for _ in range(self.n_hash_encodings)])


        elif mixing_type == 'multihead_blend_attention_style':
            n_heads = 8 # TODO expose parameter
            self.n_heads = n_heads

            self.n_features_per_head = dim_hash_encoding
            self.n_output_dims = dim_hash_encoding

            self.lin_heads = nn.ModuleList([nn.Linear(dim_hash_encoding, dim_hash_encoding//4) for _ in range(n_heads)])
            #self.lin_heads = torch.nn.ModuleList([
            #     nn.Embedding(dim_hash_encoding,
            #                                       dim_hash_encoding,
            #                                       # dtype=self.hash_encodings[0].dtype
            #                                       ).cuda() for _ in range(n_heads)
            #])
            self.lin_combine = nn.Linear(n_heads*dim_hash_encoding//4, self.n_output_dims)
            #self.lin_combine = nn.Embedding(n_heads*dim_hash_encoding,
            #                                   self.n_output_dims,
            #                                   # dtype=self.hash_encodings[0].dtype
            #                                   ).cuda()

        elif mixing_type in {'attention', 'multihead_attention'}:
            assert dim_conditioning_code is not None, "For attention mixing_type, dim_conditioning_code must be given"
            self.attention_keys = nn.Embedding(n_hash_encodings,
                                               dim_conditioning_code,
                                               # dtype=self.hash_encodings[0].dtype
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
        elif self.mixing_type == 'mlp_blend_field':
            #hidden_dim = 64
            #condition_dim = 32
            #self.n_blend_layers = 3
            #self.skips = [2]
            #self.blend_layers = nn.ModuleList(
            #    [nn.Linear(3 + condition_dim, hidden_dim)] +
            #    [nn.Linear(hidden_dim, hidden_dim) if i not in self.skips else
            #     nn.Linear(hidden_dim + 3 + condition_dim, hidden_dim) for i in range(self.n_blend_layers)]
            #)
            #self.out_layer = nn.Linear(hidden_dim, self.n_hash_encodings)
#

#

            self.extra_weight_factor = 4
            self.n_output_dims = dim_hash_encoding
            self.blend_field_config = blend_field_config
            pos_encoder, blend_field = blend_field_config.setup(dim_conditioning_code, n_hash_encodings*self.extra_weight_factor)
            self.pos_encoder = pos_encoder
            self.blend_field = blend_field

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

        if self.only_render_hash_table is not None:
            blended_embeddings = 0.06 * embeddings[:, :, self.only_render_hash_table]
        else:
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

            elif self.mixing_type in ['multihead_blend', 'multihead_blend_mixed']:
                assert conditioning_code.shape[-1] == self.n_hash_encodings * self.n_heads, \
                    "multihead_blend requires the conditioning code to have dimension n_tables * n_heads"

                B = conditioning_code.shape[0]
                C = conditioning_code.shape[1]  # code dim
                H = embeddings.shape[2]  # number of hash tables
                nH = self.n_heads
                FpH = self.n_features_per_head

                embeddings = embeddings.to(conditioning_code)
                if self.mixing_type == 'multihead_blend_mixed':
                    embeddings = torch.stack([self.mixing_heads[i](embeddings[:, :, i]) for i in range(self.n_hash_encodings)], dim=-1)

                conditioning_code = conditioning_code.repeat_interleave(FpH, dim=-1)  # [B, C * FpH], C = H * nH
                conditioning_code = conditioning_code.reshape(B, H, nH * FpH)  # [B, H, D=nH * FpH]
                conditioning_code = conditioning_code.transpose(1, 2)  # [B, D, H]
                weighted_embeddings = conditioning_code * embeddings  # [B, D, H]
                blended_embeddings = weighted_embeddings.sum(dim=2)  # [B, D]

            elif self.mixing_type == 'multihead_blend_attention_style':
                #embdeggins: B x D x H
                #conditioning_code: B x C

                B = conditioning_code.shape[0]
                C = conditioning_code.shape[1]  # code dim
                H = embeddings.shape[2]  # number of hash tables

                embeddings = embeddings.permute(0, 2, 1)  # B x H x D
                conditioning_code = conditioning_code.reshape(B, H, self.n_heads)
                embeddings = embeddings.to(conditioning_code)
                # dimension wise this works torch.einsum('jk,bnj->bnk', self.lin_heads[0].weight, embeddings) instead but casting is still a problem
                #fused_embeddings = torch.stack([ torch.matmul(self.lin_heads[i].weight, embeddings) for i in range(self.n_heads)], dim=-2) # B x H x n_heads x D
                fused_embeddings = torch.stack([ self.lin_heads[i](embeddings) for i in range(self.n_heads)], dim=-2) # B x H x n_heads x D

                scaled_f = conditioning_code.unsqueeze(-1) * fused_embeddings # B x H x n_heads x D

                scaled_f = scaled_f.sum(dim=1) # B x n_heads x D

                scaled_f = scaled_f.reshape(B, -1) # B x n_heads * D

                blended_embeddings = self.lin_combine(scaled_f) # B x D
                #blended_embeddings = torch.matmul(self.lin_combine.weight, scaled_f) # B x D
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
            elif self.mixing_type == 'mlp_blend_field':
                ## embdeggins: B x D x H
                ## conditioning_code: B x C
                #inp = torch.cat([in_tensor, conditioning_code], dim=-1)
                #hidden = inp
                #for i, layer in enumerate(self.blend_layers):
                #    hidden = layer(hidden)
                #    hidden = F.relu(hidden)
                #    if i in self.skips:
                #        hidden = torch.cat([inp, hidden], dim=-1)
#
                #weights = F.normalize(self.out_layer(hidden), dim=-1) # B x H
#
                inp = torch.cat([self.pos_encoder(in_tensor), conditioning_code], dim=-1) # B x (pos_enc_dim + C)
                B = in_tensor.shape[0]
                D = embeddings.shape[1]
                weights = self.blend_field(inp).reshape(B, self.n_hash_encodings, self.extra_weight_factor) # B x H x self.extra_weight_factor
                embeddings = embeddings.permute(0, 2, 1) # B H D
                embeddings = embeddings.reshape(B, self.n_hash_encodings, D//self.extra_weight_factor, self.extra_weight_factor)  # B x H x D//factor x factor
                blended_embeddings = (embeddings * weights.unsqueeze(2)).sum(dim=1) # B x D//self.extra_weight_factor x self.extra_weight_factor
                blended_embeddings = blended_embeddings.reshape(B, -1) # B x self.n_output_dims
            else:
                raise ValueError(f"Unsupported mixing type: {self.mixing_type}")

        return blended_embeddings

    def get_out_dim(self) -> int:
        return self.n_output_dims
