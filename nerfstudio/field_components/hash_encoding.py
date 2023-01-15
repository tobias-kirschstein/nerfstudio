import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from math import ceil, sqrt
from typing import Dict, List, Literal, Optional, Tuple

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

import einops

from nerfstudio.field_components.encodings import WindowedNeRFEncoding, posenc_window
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.hypernerf_field import SE3WarpingFieldEnsem

HashEnsembleMixingType = Literal['blend', 'attention', 'multihead_attention',
                                 'multihead_blend',
                                 'multihead_blend_mixed',
                                 'multihead_blend_attention_style',
                                 'mlp_blend_field',
                                 'multi_deform_blend',
                                 'multi_deform_blend++',
                                 'multi_deform_blend_offset'
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

    def forward(self, x, windows_param: Optional[float] = None):
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

    def setup(self, n_total_features: int) -> tcnn.Encoding:
        encoding = tcnn.Encoding(self.n_dims_to_encode, encoding_config={
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": 8 if n_total_features >= 8 else n_total_features,
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
    output_activation: Optional[Literal['Normalization', 'Tanh']] = 'Normalization'
    n_freq_pos_enc: int = 0
    skip_connections: Optional[Tuple[int]] = None
    input_dim: int = 3


    def setup(self, n_time_condition_dim: int, n_hash_tables: int) -> (WindowedNeRFEncoding, MLP):
        if self.n_freq_pos_enc > 0:
            nerf_encoder = WindowedNeRFEncoding(in_dim=self.input_dim,
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
            in_dim=n_time_condition_dim + nerf_encoder.get_out_dim() if (
                    self.n_freq_pos_enc > 0) else n_time_condition_dim + self.input_dim,
            num_layers=self.n_layers,
            layer_width=self.n_hidden_dims,
            out_dim=n_hash_tables,
            skip_connections=self.skip_connections,
            out_activation=out_activation
        )
        return nerf_encoder, mlp_blend_field


@dataclass
class MultiDeformConfig:
    n_hidden_dims: int = 64
    n_layers: int = 4
    # output_activation: Optional[Literal['Normalization', 'Tanh']] = 'Normalization'
    n_freq_pos_enc: int = 0
    skip_connections: Optional[Tuple[int]] = None
    input_dim: int = 3
    blend_weight_dim: int = 1

    def setup(self, n_time_condition_dim: int, n_hash_tables: int) -> (WindowedNeRFEncoding, MLP):
        if self.n_freq_pos_enc > 0:
            nerf_encoder = WindowedNeRFEncoding(in_dim=self.input_dim,
                                                num_frequencies=self.n_freq_pos_enc,
                                                min_freq_exp=0,
                                                max_freq_exp=self.n_freq_pos_enc - 1,
                                                include_input=True)
        else:
            nerf_encoder = IdenEnc()
        # if self.output_activation is None:
        #    out_activation = None
        # elif self.output_activation == 'Normalization':
        #    out_activation = Normalizer()
        # elif self.output_activation == 'Tanh':
        #    out_activation = nn.Tanh()
        mlp_blend_field = MLP(
            in_dim=n_time_condition_dim + nerf_encoder.get_out_dim() if (
                    self.n_freq_pos_enc > 0) else n_time_condition_dim + self.input_dim,
            num_layers=self.n_layers,
            layer_width=self.n_hidden_dims,
            out_dim=n_hash_tables * (self.input_dim + self.blend_weight_dim),
            skip_connections=self.skip_connections,
            out_activation=None  # out_activation
        )
        return nerf_encoder, mlp_blend_field


@dataclass
class MultiDeformSE3Config:
    n_hidden_dims: int = 128
    n_layers: int = 6
    n_freq_pos_enc: int = 7

    def setup(self, n_time_condition_dim: int, n_hash_tables: int) -> SE3WarpingFieldEnsem:
        se3_warp_field = SE3WarpingFieldEnsem(
            n_freq_pos=self.n_freq_pos_enc,
            mlp_num_layers=self.n_layers,
            mlp_layer_width=self.n_hidden_dims,
            warp_code_dim=n_time_condition_dim,
            warp_direction=False,
            n_output_deformations=n_hash_tables
        )
        return se3_warp_field


class HashEncodingEnsemble(nn.Module):

    def __init__(self,
                 n_hash_encodings: int,
                 hash_encoding_config: TCNNHashEncodingConfig,
                 mixing_type: HashEnsembleMixingType = 'blend',
                 dim_conditioning_code: Optional[int] = None,
                 n_heads: Optional[int] = None,
                 only_render_hash_table: Optional[int] = None,
                 blend_field_config: BlendFieldConfig = None,
                 multi_deform_config: MultiDeformConfig = None,
                 multi_deform_se3_config: MultiDeformSE3Config = None,
                 disable_initial_hash_ensemble: bool = False,
                 disable_table_chunking: bool = False,
                 use_soft_transition: bool = False):
        super(HashEncodingEnsemble, self).__init__()

        self.mixing_type = mixing_type
        self.n_hash_encodings = n_hash_encodings
        self.hash_encoding_config = hash_encoding_config
        self.only_render_hash_table = only_render_hash_table
        self.disable_initial_hash_ensemble = disable_initial_hash_ensemble
        self.disable_table_chunking = disable_table_chunking
        self.use_soft_transition = use_soft_transition

        if mixing_type in {'multi_deform_blend', 'multi_deform_blend_offset'} or disable_table_chunking:
            # Multi-deform mixing types cannot chunk the hash tables as the deformed inputs vary for every hash table
            self.hash_encodings = []
            for i_hash_encoding in range(n_hash_encodings):
                self.hash_encodings.append(hash_encoding_config.setup(hash_encoding_config.n_features_per_level))

            self.hash_encodings = nn.ModuleList(self.hash_encodings)
        else:
            n_total_features = n_hash_encodings * hash_encoding_config.n_features_per_level
            assert n_total_features <= 8 \
                   or n_total_features % 8 == 0, \
                "Number of features in hashtables must either be smaller than 8 or a multiple of 8!"
            self.hash_encodings = []
            for i_hash_encoding in range(ceil(n_total_features / 8)):
                self.hash_encodings.append(hash_encoding_config.setup(n_total_features))

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
            n_heads = 8  # TODO expose parameter
            self.n_heads = n_heads

            self.n_features_per_head = dim_hash_encoding
            self.n_output_dims = dim_hash_encoding

            self.lin_heads = nn.ModuleList(
                [nn.Linear(dim_hash_encoding, dim_hash_encoding // 4) for _ in range(n_heads)])

            self.lin_combine = nn.Linear(n_heads * dim_hash_encoding // 4, self.n_output_dims)

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
        elif self.mixing_type == 'mlp_blend_field' or self.mixing_type in ['multi_deform_blend', 'multi_deform_blend++']:
            self.extra_weight_factor = 4
            self.n_output_dims = dim_hash_encoding
            self.blend_field_config = blend_field_config
            pos_encoder, blend_field = blend_field_config.setup(dim_conditioning_code,
                                                                n_hash_encodings * self.extra_weight_factor)
            self.pos_encoder = pos_encoder
            self.blend_field = blend_field

        elif self.mixing_type == 'blend':
            assert n_hash_encodings == dim_conditioning_code, 'For simple blend n_hash_encodings has to equal dim_conditioning_code (which is latent_dim_time)'

        if self.mixing_type in ['multi_deform_blend_offset', 'multi_deform_blend', 'multi_deform_blend++']:
            self.n_output_dims = dim_hash_encoding

            if self.mixing_type == 'multi_deform_blend++':
                dim_conditioning_code -= self.n_hash_encodings * self.extra_weight_factor

            if self.mixing_type == 'multi_deform_blend_offset':
                self.multi_deform_config = multi_deform_config
                pos_encoder, multi_deform_mlp = self.multi_deform_config.setup(dim_conditioning_code, n_hash_encodings)
                self.pos_encoder = pos_encoder
                self.multi_deform_mlp = multi_deform_mlp

            else:
                self.multi_deform_se3_field = multi_deform_se3_config.setup(dim_conditioning_code, n_hash_encodings)

        else:
            self.n_output_dims = dim_hash_encoding

        # Unfortunately, cannot merge hashtables into a single hashtable, as the maximum number of features_per_level is 8!
        # hash_encoding_config_merged = dataclasses.replace(hash_encoding_config,
        #                                                   n_features_per_level=hash_encoding_config.n_features_per_level * n_hash_encodings)
        # self.hash_encoding = hash_encoding_config_merged.setup()

    def forward(self,
                in_tensor: torch.Tensor,
                conditioning_code: torch.Tensor,
                windows_param: Optional[float] = None,
                windows_param_blend_field: Optional[float] = None,
                windows_param_tables: Optional[float] = None,
                windows_param_deform: Optional[float] = None) -> torch.Tensor:

        B = in_tensor.shape[0]

        # deform query positions for each hash table in multi_deform_mode
        # (multi-deform_mlp also gives blend weights)
        if self.mixing_type in ['multi_deform_blend', 'multi_deform_blend++', 'multi_deform_blend_offset']:
            if self.mixing_type == 'multi_deform_blend_offset':
                encoded_xyz = self.pos_encoder(in_tensor, windows_param=windows_param_blend_field)
                inp_multi_deform = torch.cat([encoded_xyz, conditioning_code], dim=-1)
                offsets_and_weights = self.multi_deform_mlp(inp_multi_deform).view(B, self.n_hash_encodings,
                                                                                   -1)  # B x H x (3 + weight_dim)
                offsets = offsets_and_weights[:, :, :self.multi_deform_config.input_dim]
                blend_weights = offsets_and_weights[:, :,
                                self.multi_deform_config.input_dim]  # for now only use one weight dim !!!!
                in_tensor_deformed = in_tensor.unsqueeze(1) + offsets
            else:
                warped_positions, _ = self.multi_deform_se3_field(
                    in_tensor,
                    warp_code=conditioning_code if self.mixing_type == 'multi_deform_blend'
                        else conditioning_code[..., :-self.n_hash_encodings*self.extra_weight_factor],
                    windows_param=windows_param_deform)

                in_tensor_deformed = warped_positions  # [B, H, 3]

            embeddings = []
            for h, hash_encoding in enumerate(self.hash_encodings):
                embedding = hash_encoding(in_tensor_deformed[:, h, :])
                embeddings.append(embedding)

            embeddings = torch.stack(embeddings, dim=-1)  # [B, D, H]
        else:
            if self.disable_table_chunking:
                # backwards compatibility
                embeddings = []
                for h, hash_encoding in enumerate(self.hash_encodings):
                    embedding = hash_encoding(in_tensor)
                    embeddings.append(embedding)

                embeddings = torch.stack(embeddings, dim=-1)  # [B, D, H]
            else:

                embeddings = []
                for h, hash_encoding in enumerate(self.hash_encodings):
                    embedding = hash_encoding(in_tensor)
                    embeddings.append(embedding)

                embeddings = torch.stack(embeddings, dim=1)  # [B, C, 8 * L]
                C = embeddings.shape[1]
                L = self.hash_encoding_config.n_levels
                F = self.hash_encoding_config.n_features_per_level
                P = int(8 / F) if F * self.n_hash_encodings >= 8 else self.n_hash_encodings

                #embeddings = embeddings.reshape((B, C, L, P, F))
                #embeddings = embeddings.transpose(2, 3)  # [B, C, P, L, F]
                #embeddings = embeddings.reshape((B, C*P, L*F))
                #embeddings = embeddings.transpose(1, 2)  # [B, D, H]

                # ordering of features might be slightly different, before features from one level were next to each other (?)
                # now the first features of each level are next to each other then the second features across all level etc.
                embeddings = einops.rearrange(embeddings, 'b c (l p f) -> b (f l) (c p)', l=L, p=P, f=F)
        if windows_param_tables is not None:
            # Gradually add more tables

            if windows_param_tables == 1 and self.disable_initial_hash_ensemble:
                # Force deformation network to learn correspondences as long as only one table is active
                conditioning_code = torch.ones_like(conditioning_code)
            elif self. use_soft_transition and windows_param_tables < 2:
                # Slowly migrate to using the actual conditioning code instead of fixing the blend weights to 1
                alpha = windows_param_tables - 1  # Goes from 0 -> 1

                if self.mixing_type == 'blend':
                    # Only first entry of conditioning code is responsible for first table
                    conditioning_code[:, 0] = alpha * conditioning_code[:, 0] + (1 - alpha) * 1
                elif self.mixing_type == 'multihead_blend':
                    # First n_heads entries of conditioning code are responsible for first table
                    conditioning_code[:, :self.n_heads] = alpha * conditioning_code[:, :self.n_heads] \
                                                       + (1 - alpha) * torch.ones_like(conditioning_code[:, :self.n_heads])
                else:
                    raise NotImplementedError("slow_migration only implemented for mixing types blend and multihead_blend")

            window = posenc_window(windows_param_tables,
                                   0,
                                   self.n_hash_encodings - 1,
                                   self.n_hash_encodings)  # [H]
            window = window.unsqueeze(0).unsqueeze(1).to(embeddings)  # [1, 1, H]
            embeddings = window * embeddings

        if windows_param is not None:
            # Gradually add higher frequency detail
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

                C = conditioning_code.shape[1]  # code dim
                H = embeddings.shape[2]  # number of hash tables
                nH = self.n_heads
                FpH = self.n_features_per_head

                embeddings = embeddings.to(conditioning_code)
                if self.mixing_type == 'multihead_blend_mixed':
                    embeddings = torch.stack(
                        [self.mixing_heads[i](embeddings[:, :, i]) for i in range(self.n_hash_encodings)], dim=-1)

                conditioning_code = conditioning_code.repeat_interleave(FpH, dim=-1)  # [B, C * FpH], C = H * nH
                conditioning_code = conditioning_code.reshape(B, H, nH * FpH)  # [B, H, D=nH * FpH]
                conditioning_code = conditioning_code.transpose(1, 2)  # [B, D, H]
                weighted_embeddings = conditioning_code * embeddings  # [B, D, H]
                blended_embeddings = weighted_embeddings.sum(dim=2)  # [B, D]

            elif self.mixing_type == 'multihead_blend_attention_style':
                # embdeggins: B x D x H
                # conditioning_code: B x C

                C = conditioning_code.shape[1]  # code dim
                H = embeddings.shape[2]  # number of hash tables

                embeddings = embeddings.permute(0, 2, 1)  # B x H x D
                conditioning_code = conditioning_code.reshape(B, H, self.n_heads)
                embeddings = embeddings.to(conditioning_code)
                # dimension wise this works torch.einsum('jk,bnj->bnk', self.lin_heads[0].weight, embeddings) instead but casting is still a problem
                # fused_embeddings = torch.stack([ torch.matmul(self.lin_heads[i].weight, embeddings) for i in range(self.n_heads)], dim=-2) # B x H x n_heads x D
                fused_embeddings = torch.stack([self.lin_heads[i](embeddings) for i in range(self.n_heads)],
                                               dim=-2)  # B x H x n_heads x D

                scaled_f = conditioning_code.unsqueeze(-1) * fused_embeddings  # B x H x n_heads x D

                scaled_f = scaled_f.sum(dim=1)  # B x n_heads x D

                scaled_f = scaled_f.reshape(B, -1)  # B x n_heads * D

                blended_embeddings = self.lin_combine(scaled_f)  # B x D
                # blended_embeddings = torch.matmul(self.lin_combine.weight, scaled_f) # B x D
            elif self.mixing_type == 'attention':
                # Scaled dot-product attention
                keys = self.attention_keys.weight  # [H, T]
                queries = conditioning_code  # [B, T]
                values = embeddings  # [B, D, H]

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
            elif self.mixing_type == 'mlp_blend_field' or self.mixing_type in ['multi_deform_blend',
                                                                               'multi_deform_blend++']:
                ## embeddins: B x D x H
                ## conditioning_code: B x C
                encoded_xyz = self.pos_encoder(in_tensor, windows_param=windows_param_blend_field)
                inp = torch.cat([encoded_xyz, conditioning_code], dim=-1)  # B x (pos_enc_dim + C)
                D = embeddings.shape[1]
                weights = self.blend_field(inp).reshape(B, self.n_hash_encodings,
                                                        self.extra_weight_factor)  # B x H x self.extra_weight_factor
                if self.mixing_type == 'multi_head_blend++':
                    # add directly optimized blend weights as correctives
                    weight_correctives = conditioning_code[..., -self.n_hash_encodings*self.extra_weight_factor:].reshape(B, self.n_hash_encodings,
                                                        self.extra_weight_factor)  # B x H x self.extra_weight_factor
                    weights += weight_correctives

                embeddings = embeddings.permute(0, 2, 1)  # B H D
                embeddings = embeddings.reshape(B, self.n_hash_encodings, D // self.extra_weight_factor,
                                                self.extra_weight_factor)  # B x H x D//factor x factor
                blended_embeddings = (embeddings * weights.unsqueeze(2)).sum(
                    dim=1)  # B x D//self.extra_weight_factor x self.extra_weight_factor
                blended_embeddings = blended_embeddings.reshape(B, -1)  # B x self.n_output_dims

            elif self.mixing_type == 'multi_deform_blend_offset':
                assert self.multi_deform_config.blend_weight_dim == 1, "multi-level blending not implemented yet"
                # curretnly blend_weights: B x H
                # embeddings B x D x H
                blended_embeddings = (blend_weights.unsqueeze(1) * embeddings).sum(dim=-1)

            else:
                raise ValueError(f"Unsupported mixing type: {self.mixing_type}")

        return blended_embeddings

    def get_out_dim(self) -> int:
        return self.n_output_dims

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = defaultdict(list)

        param_groups["fields"] = list(self.hash_encodings.parameters())

        if self.mixing_type == 'mlp_blend_field':
            param_groups["blend_fields"] = list(self.blend_field.parameters())
        elif self.mixing_type in ['multi_deform_blend', 'multi_deform_blend++']:
            param_groups["blend_fields"] = list(self.blend_field.parameters())
            param_groups["blend_fields"] = list(self.multi_deform_se3_field.parameters())
        elif self.mixing_type == 'multi_deform_blend_offset':
            param_groups["blend_fields"] = list(self.multi_deform_mlp.parameters())

        return param_groups

class HashEncodingEnsembleParallel(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        n_input_dims: int = 3,
        base_resolution: int = 16,
        n_levels: int = 14,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        per_level_scale: float = 1.4472692012786865,
    ):
        super(HashEncodingEnsembleParallel, self).__init__()

        self.ensemble_size = ensemble_size
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.n_output_dims = n_levels * n_features_per_level

        self.encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level * ensemble_size,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            },
        )

    def forward(
        self,
        x: torch.Tensor,
        ensemble_code: torch.Tensor,
    ) -> torch.Tensor:
        flattened_feat = self.encoding(x)
        stacked_feat = flattened_feat.reshape(
            -1, self.n_levels, self.ensemble_size, self.n_features_per_level
        )  # (B*S, L, E, F)
        weighted_feat = ensemble_code[:, None, :, None] * stacked_feat
        feat = weighted_feat.sum(dim=-2).reshape(flattened_feat.shape[0], -1)
        return feat

    def get_out_dim(self) -> int:
        return self.n_output_dims
