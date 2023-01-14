from typing import Optional, Iterable

from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import WindowedNeRFEncoding
from nerfstudio.fields.hypernerf_field import SE3WarpingField

try:
    import pytorch3d
    import pytorch3d.transforms
except RuntimeError as e:
    print("pytorch3d not installed => SE3Field is not available. Check their github page for installation instructions!")

import tinycudann as tcnn
import torch
from torch import nn
from torch.nn import init


def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
      w: (B, 3,) A 3-vector
    Returns:
      W: (B, 3, 3) A skew matrix such that W @ v == w x v
    """
    B = w.shape[0]
    assert len(w.shape) == 2
    W = torch.zeros((B, 3, 3), dtype=w.dtype, device=w.device)

    # torch.tensor([[0.0, -w[2], w[1]],
    #                          [w[2], 0.0, -w[0]],
    #                          [-w[1], w[0], 0.0]])

    W[:, 0, 1] = -w[:, 2]
    W[:, 0, 2] = w[:, 1]
    W[:, 1, 0] = w[:, 2]
    W[:, 1, 2] = -w[:, 0]
    W[:, 2, 0] = -w[:, 1]
    W[:, 2, 1] = w[:, 0]
    return W


def exp_so3(W: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
    Args:
      W: skew symmetric matrix W (B, 3, 3) derived from a (B, 3,) axis of rotation.
        Note, it is assumed that the w that W is derived from is a unit vector
      theta: (B,) An angle of rotation.
    Returns:
      R: (B, 3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    # W = skew(w)
    assert W.shape[0] == theta.shape[0]
    B = W.shape[0]
    theta = theta.view(B, 1, 1)
    identities = torch.eye(3, dtype=W.dtype, device=W.device).unsqueeze(0).repeat((B, 1, 1))
    # Here, it is assumed that W is derived from unit vector rotation axes w.
    # Otherwise, would have to divide by theta and theta^2
    return identities + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W @ W
    # return identities + torch.sin(theta) / theta * W + (1.0 - torch.cos(theta) / theta.pow(2)) * W @ W


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.
    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.
    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    # p = p.reshape((3, 1))
    assert R.shape[0] == p.shape[0]
    B = R.shape[0]
    upper_transform = torch.concat([R, p], dim=-1)  # [3, 4]
    homogeneous_vectors = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(
        (B, 1, 1))
    total_transform = torch.concat([upper_transform,
                                    homogeneous_vectors], dim=1)

    return total_transform


def exp_se3(S: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
      S: (6,) A screw axis of motion. Consists of 2 3-vectors with unit norm
      theta: Magnitude of motion.
    Returns:
      a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    assert S.shape[0] == theta.shape[0]
    B = S.shape[0]

    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(W, theta)

    theta = theta.view(B, 1, 1)
    identities = torch.eye(3, dtype=S.dtype, device=S.device).unsqueeze(0).repeat((B, 1, 1))
    p = (theta * identities + (1.0 - torch.cos(theta)) * W +
         (theta - torch.sin(theta)) * W @ W) @ v.unsqueeze(-1)
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]


class DeformationField(nn.Module):
    def __init__(self, n_hidden_layers: int, hidden_dim: int, latent_dim_time: int, n_freq_pos=7):
        super(DeformationField, self).__init__()
        # self.deformation_network = tcnn.Network(
        #     n_input_dims=3 + latent_dim_time,
        #     n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": n_hidden_layers,
        #     },
        # )

        self.position_encoding = WindowedNeRFEncoding(
            in_dim=3, num_frequencies=n_freq_pos, min_freq_exp=0.0, max_freq_exp=n_freq_pos - 1, include_input=True
        )
        # self.deformation_network = MLP(self.position_encoding.get_out_dim() + latent_dim_time,
        #                                n_hidden_layers,
        #                                hidden_dim,
        #                                3,
        #                                activation=nn.ReLU())
        self.deformation_network = SE3WarpingField(warp_code_dim=latent_dim_time)

    def _backward_hook(self, module, grad_input, grad_output):
        # Important if one wants to set a breakpoint inside the backward hook
        try:
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except ModuleNotFoundError:
            pass

        return grad_input

    def _tensor_hook(self, g):
        try:
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except ModuleNotFoundError:
            pass

        print(f"Tensor hook: {g}")

        # Downscale gradients
        g = 0.1 * g

        return g

    def forward(self, points: torch.Tensor, latent_codes: torch.Tensor, windows_param: Optional[float] = None) -> torch.Tensor:

        # encoded_xyz = self.position_encoding(points, windows_param=windows_param)  # [B, F]

        # network_input = torch.concat([encoded_xyz, latent_codes], dim=1)  # [B, F + D]
        warped_points = self.deformation_network(points, warp_code=latent_codes, windows_param=windows_param).to(points)

        return warped_points


class SE3Field(nn.Module):
    """
    Models deformations as SE3 transformations per 3D point.
    x' = D(x)
    outputs are already deformed points (not deformation offset)
    """

    def __init__(self,
                 dim_latent_code: int,
                 hidden_dim: int,
                 n_layers_trunk=3,
                 n_layers_w_head=3,
                 n_layers_v_head=3,
                 n_frequencies: int = 4,
                 n_output_deformations: int = 1):
        super(SE3Field, self).__init__()

        self.n_output_deformations = n_output_deformations
        n_input_dims_trunk = 3 + dim_latent_code
        self.trunk = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims_trunk,
            n_output_dims=hidden_dim,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "n_frequencies": n_frequencies
                    },
                    {
                        "otype": "Identity"  # Number of remaining input dimensions is automatically derived
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_layers_trunk,
            },
        )

        n_input_dims_w_head = hidden_dim
        self.w_head = tcnn.Network(
            n_input_dims=n_input_dims_w_head,
            n_output_dims=3 * n_output_deformations,
            network_config={
                "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_layers_w_head,
            }
        )

        n_input_dims_v_head = hidden_dim
        self.v_head = tcnn.Network(
            n_input_dims=n_input_dims_v_head,
            n_output_dims=3 * n_output_deformations,
            network_config={
                "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_layers_v_head,
            }
        )

        # TODO: Only initialize the last layer with 1e-4
        # TODO: Maybe use uniform initilization
        init.normal_(self.w_head.params, 0, 1e-4)
        init.normal_(self.v_head.params, 0, 1e-3)
        init.normal_(self.trunk.params, 0, 0.1)

    def forward(self, points: torch.Tensor, latent_codes: torch.Tensor):
        assert points.shape[0] == latent_codes.shape[0]
        assert len(points.shape) == 2
        assert len(latent_codes.shape) == 2

        B = points.shape[0]
        O = self.n_output_deformations

        trunk_inputs = torch.concat([points, latent_codes], dim=1)
        trunk_output = self.trunk(trunk_inputs)  # [B, D]

        # TODO: maximum scale is somewhat arbitrary
        # w = (self.w_head(trunk_output) - 0.5) * 4 * np.pi  # [B, 3]
        # v = (self.v_head(trunk_output) - 0.5) * 2 # [B, 3]
        w = self.w_head(trunk_output)  # [B, 3 * O]
        v = self.v_head(trunk_output)  # [B, 3 * O]

        # theta = w.norm(dim=1, p=2)
        # w = w / theta[:, None]
        # v = v / theta[:, None]
        #
        # screw_axis = torch.concat([w, v], dim=1)  # [B, 6]
        # transforms = exp_se3(screw_axis, theta).to(points.dtype)

        screw_axis = torch.stack([v, w], dim=-1)  # [B, 3*O, 2]
        screw_axis = screw_axis.reshape((B * O, 3, 2))  # [B*O, 3, 2]
        screw_axis = screw_axis.transpose(1, 2)  # [B*O, 2, 3]
        screw_axis = screw_axis.reshape((B*O, 6))  # [B*O, 6]

        # screw_axis = torch.concat([v, w], dim=1)  # [B, 6]
        transforms = pytorch3d.transforms.se3_exp_map(screw_axis).to(points.dtype)  # [B * O, 4, 4]
        transforms = transforms.permute(0, 2, 1)  # Not sure why this permute is necessary

        warped_points = from_homogenous((transforms @ to_homogenous(points).unsqueeze(-1)).squeeze(-1))  # [B * O, 3]
        warped_points = warped_points.reshape((B, O, 3))  # [B, O, 3]
        warped_points = warped_points.squeeze(1)  # [B, 3] in case only one output deformation was requested
        warped_points = warped_points.to(points.dtype)
        idx_nan = warped_points.isnan()
        warped_points[idx_nan] = points[idx_nan]  # If deformation is NaN, just use original point

        assert len(warped_points.shape) == 2
        assert warped_points.shape[0] == B
        assert warped_points.shape[1] == 3

        return warped_points.to(points.dtype)


def _nerfies_warp_field():
    class SE3Field(nn.Module):
        """Network that predicts warps as an SE(3) field.
        Attributes:
          points_encoder: the positional encoder for the points.
          metadata_encoder: an encoder for metadata.
          alpha: the alpha for the positional encoding.
          skips: the index of the layers with skip connections.
          depth: the depth of the network excluding the logit layer.
          hidden_channels: the width of the network hidden layers.
          activation: the activation for each layer.
          metadata_encoded: whether the metadata parameter is pre-encoded or not.
          hidden_initializer: the initializer for the hidden layers.
          output_initializer: the initializer for the last logit layer.
        """
        num_freqs: int
        num_embeddings: int
        num_embedding_features: int
        min_freq_log2: int = 0
        max_freq_log2: Optional[int] = None
        use_identity_map: bool = True

        activation: types.Activation = nn.relu
        skips: Iterable[int] = (4,)
        trunk_depth: int = 6
        trunk_width: int = 128
        rotation_depth: int = 0
        rotation_width: int = 128
        pivot_depth: int = 0
        pivot_width: int = 128
        translation_depth: int = 0
        translation_width: int = 128
        metadata_encoder_type: str = 'glo'
        metadata_encoder_num_freqs: int = 1

        default_init: types.Initializer = nn.initializers.xavier_uniform()
        rotation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
        pivot_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
        translation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)

        use_pivot: bool = False
        use_translation: bool = False

        def setup(self):
            self.points_encoder = modules.AnnealedSinusoidalEncoder(
                num_freqs=self.num_freqs,
                min_freq_log2=self.min_freq_log2,
                max_freq_log2=self.max_freq_log2,
                use_identity=self.use_identity_map)

            if self.metadata_encoder_type == 'glo':
                self.metadata_encoder = glo.GloEncoder(
                    num_embeddings=self.num_embeddings,
                    features=self.num_embedding_features)
            elif self.metadata_encoder_type == 'time':
                self.metadata_encoder = modules.TimeEncoder(
                    num_freqs=self.metadata_encoder_num_freqs,
                    features=self.num_embedding_features)
            else:
                raise ValueError(
                    f'Unknown metadata encoder type {self.metadata_encoder_type}')

            self.trunk = modules.MLP(
                depth=self.trunk_depth,
                width=self.trunk_width,
                hidden_activation=self.activation,
                hidden_init=self.default_init,
                skips=self.skips)

            branches = {
                'w':
                    modules.MLP(
                        depth=self.rotation_depth,
                        width=self.rotation_width,
                        hidden_activation=self.activation,
                        hidden_init=self.default_init,
                        output_init=self.rotation_init,
                        output_channels=3),
                'v':
                    modules.MLP(
                        depth=self.pivot_depth,
                        width=self.pivot_width,
                        hidden_activation=self.activation,
                        hidden_init=self.default_init,
                        output_init=self.pivot_init,
                        output_channels=3),
            }
            if self.use_pivot:
                branches['p'] = modules.MLP(
                    depth=self.pivot_depth,
                    width=self.pivot_width,
                    hidden_activation=self.activation,
                    hidden_init=self.default_init,
                    output_init=self.pivot_init,
                    output_channels=3)
            if self.use_translation:
                branches['t'] = modules.MLP(
                    depth=self.translation_depth,
                    width=self.translation_width,
                    hidden_activation=self.activation,
                    hidden_init=self.default_init,
                    output_init=self.translation_init,
                    output_channels=3)
            # Note that this must be done this way instead of using mutable operations.
            # See https://github.com/google/flax/issues/524.
            self.branches = branches

        def encode_metadata(self,
                            metadata: jnp.ndarray,
                            time_alpha: Optional[float] = None):
            if self.metadata_encoder_type == 'time':
                metadata_embed = self.metadata_encoder(metadata, time_alpha)
            elif self.metadata_encoder_type == 'glo':
                metadata_embed = self.metadata_encoder(metadata)
            else:
                raise RuntimeError(
                    f'Unknown metadata encoder type {self.metadata_encoder_type}')

            return metadata_embed

        def warp(self,
                 points: jnp.ndarray,
                 metadata_embed: jnp.ndarray,
                 extra: Dict[str, Any]):
            points_embed = self.points_encoder(points, alpha=extra.get('alpha'))
            inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
            trunk_output = self.trunk(inputs)

            w = self.branches['w'](trunk_output)
            v = self.branches['v'](trunk_output)
            theta = jnp.linalg.norm(w, axis=-1)
            w = w / theta[..., jnp.newaxis]
            v = v / theta[..., jnp.newaxis]
            screw_axis = jnp.concatenate([w, v], axis=-1)
            transform = rigid.exp_se3(screw_axis, theta)

            warped_points = points
            if self.use_pivot:
                pivot = self.branches['p'](trunk_output)
                warped_points = warped_points + pivot

            warped_points = rigid.from_homogenous(
                transform @ rigid.to_homogenous(warped_points))

            if self.use_pivot:
                warped_points = warped_points - pivot

            if self.use_translation:
                t = self.branches['t'](trunk_output)
                warped_points = warped_points + t

            return warped_points

        def __call__(self,
                     points: jnp.ndarray,
                     metadata: jnp.ndarray,
                     extra: Dict[str, Any],
                     return_jacobian: bool = False,
                     metadata_encoded: bool = False):
            """Warp the given points using a warp field.
            Args:
              points: the points to warp.
              metadata: metadata indices if metadata_encoded is False else pre-encoded
                metadata.
              extra: A dictionary containing
                'alpha': the alpha value for the positional encoding.
                'time_alpha': the alpha value for the time positional encoding
                  (if applicable).
              return_jacobian: if True compute and return the Jacobian of the warp.
              metadata_encoded: if True assumes the metadata is already encoded.
            Returns:
              The warped points and the Jacobian of the warp if `return_jacobian` is
                True.
            """
            if metadata_encoded:
                metadata_embed = metadata
            else:
                metadata_embed = self.encode_metadata(metadata, extra.get('time_alpha'))

            out = {'warped_points': self.warp(points, metadata_embed, extra)}

            if return_jacobian:
                jac_fn = jax.jacfwd(self.warp, argnums=0)
                out['jacobian'] = jac_fn(points, metadata_embed, extra)

            return
