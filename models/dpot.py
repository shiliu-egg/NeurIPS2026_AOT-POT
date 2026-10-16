# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F



import math
import logging
from torch.nn.modules.container import Sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


################################################################################
# ComplexGroupNorm: Normalize real and imaginary parts separately
# Used in AFNO2D between the two frequency-domain MLP layers
################################################################################

class ComplexGroupNorm(nn.Module):
    """
    GroupNorm for complex-valued tensors in frequency domain.
    Normalizes real and imaginary parts separately.

    Used in AFNO2D between the two MLP layers to stabilize training.
    """
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.norm_real = nn.GroupNorm(num_groups, num_channels)
        self.norm_imag = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: (B, H, W, num_blocks, block_size) real part
            x_imag: (B, H, W, num_blocks, block_size) imaginary part
        Returns:
            Normalized (x_real, x_imag) with same shapes
        """
        B, H, W, num_blocks, block_size = x_real.shape

        # Reshape to (B, C, H, W) for GroupNorm where C = num_blocks * block_size
        x_real = x_real.permute(0, 3, 4, 1, 2).reshape(B, num_blocks * block_size, H, W)
        x_imag = x_imag.permute(0, 3, 4, 1, 2).reshape(B, num_blocks * block_size, H, W)

        # Apply GroupNorm
        x_real = self.norm_real(x_real)
        x_imag = self.norm_imag(x_imag)

        # Reshape back to (B, H, W, num_blocks, block_size)
        x_real = x_real.reshape(B, num_blocks, block_size, H, W).permute(0, 3, 4, 1, 2)
        x_imag = x_imag.reshape(B, num_blocks, block_size, H, W).permute(0, 3, 4, 1, 2)

        return x_real, x_imag


################################################################################
# mHC (Manifold-Constrained Hyper-Connections) Components
# Based on: "mHC: Manifold-Constrained Hyper-Connections" by DeepSeek-AI
################################################################################

class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm: Projects a matrix onto the Birkhoff polytope (doubly stochastic manifold).

    A doubly stochastic matrix has:
    - All non-negative entries
    - Each row sums to 1
    - Each column sums to 1
    - Spectral norm ≤ 1 (ensures stability)

    The algorithm alternates between row and column normalization until convergence.

    Args:
        num_iter: Number of Sinkhorn iterations (paper uses 20)
        epsilon: Small constant for numerical stability
    """
    def __init__(self, num_iter=20, epsilon=1e-8):
        super().__init__()
        self.num_iter = num_iter
        self.epsilon = epsilon

    def forward(self, x):
        """
        Args:
            x: (..., n, n) arbitrary real-valued matrix
        Returns:
            (..., n, n) doubly stochastic matrix
        """
        # Clamp to prevent exp overflow (exp(30) ≈ 1e13, exp(-30) ≈ 1e-13)
        x = torch.clamp(x, min=-30.0, max=30.0)

        # Exponentiate to ensure non-negativity
        M = torch.exp(x)

        # Alternating row and column normalization
        for _ in range(self.num_iter):
            # Row normalization: each row sums to 1
            M = M / (M.sum(dim=-1, keepdim=True) + self.epsilon)
            # Column normalization: each column sums to 1
            M = M / (M.sum(dim=-2, keepdim=True) + self.epsilon)

        return M


class MHCMapping(nn.Module):
    """
    mHC Mapping Computation Module - Implements the dynamic mapping mechanism from the paper.

    Computes three mapping matrices: H^pre, H^post, H^res
    - H^pre: Aggregation weights (n streams → 1 input)
    - H^post: Distribution weights (1 output → n streams)
    - H^res: Residual mixing matrix (n×n doubly stochastic)

    Each mapping consists of:
    - Dynamic component: α · (x' · φ) where x' is RMSNorm(flatten(x_stream))
    - Static component: bias b

    Args:
        n_streams: Number of parallel streams n (paper uses 4)
        hidden_dim: Single stream hidden dimension C (DPOT Tiny: 512)
        sinkhorn_iter: Number of Sinkhorn iterations (paper uses 20)
        alpha_init: Initial value for gating scalars (paper uses 0.01)
    """
    def __init__(self, n_streams, hidden_dim, sinkhorn_iter=20, alpha_init=0.01):
        super().__init__()
        self.n_streams = n_streams
        self.hidden_dim = hidden_dim
        total_dim = n_streams * hidden_dim  # nC

        # RMSNorm for the flattened nC-dimensional vector
        # Using LayerNorm without affine as a simple RMSNorm approximation
        # The paper absorbs RMSNorm weights into φ
        self.rms_norm = nn.LayerNorm(total_dim, elementwise_affine=False)

        # Dynamic mapping linear projections φ
        # φ^pre, φ^post ∈ R^{nC × n}, φ^res ∈ R^{nC × n²}
        self.phi_pre = nn.Linear(total_dim, n_streams, bias=False)
        self.phi_post = nn.Linear(total_dim, n_streams, bias=False)
        self.phi_res = nn.Linear(total_dim, n_streams * n_streams, bias=False)

        # Static biases b
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams, n_streams))

        # Gating scalars α (initialized small so dynamic component starts weak)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

        # Sinkhorn projector for H^res
        self.sinkhorn = SinkhornKnopp(num_iter=sinkhorn_iter)

        # Initialize φ with small values (combined with small α ensures stability at init)
        nn.init.normal_(self.phi_pre.weight, std=0.02)
        nn.init.normal_(self.phi_post.weight, std=0.02)
        nn.init.normal_(self.phi_res.weight, std=0.02)

        # Initialize b_res to identity matrix (so initial H^res ≈ identity after Sinkhorn)
        # This ensures the residual connection initially behaves like standard residual
        with torch.no_grad():
            self.b_res.copy_(torch.eye(n_streams))

    def forward(self, x_stream):
        """
        Args:
            x_stream: (B, n, C, H, W) multi-stream hidden state

        Returns:
            H_pre: (B, n) aggregation weights (will be normalized before use)
            H_post: (B, n) distribution weights
            H_res: (B, n, n) residual mixing matrix (doubly stochastic)
        """
        B, n, C, H, W = x_stream.shape

        # Step 1: Spatial pooling + Flatten
        # (B, n, C, H, W) → (B, n, C) → (B, nC)
        x_pooled = x_stream.mean(dim=(-2, -1))  # Global average pooling over spatial dims
        x_flat = x_pooled.reshape(B, -1)  # (B, nC)

        # Step 2: RMSNorm
        x_norm = self.rms_norm(x_flat)  # (B, nC)

        # Step 3: Compute dynamic mappings + static biases
        # H̃ = α · (x' · φ) + b
        H_tilde_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre      # (B, n)
        H_tilde_post = self.alpha_post * self.phi_post(x_norm) + self.b_post  # (B, n)
        H_tilde_res = self.alpha_res * self.phi_res(x_norm).view(B, n, n) + self.b_res  # (B, n, n)

        # Step 4: Constraint projections
        H_pre = torch.sigmoid(H_tilde_pre)        # (B, n), non-negative via sigmoid
        H_post = 2 * torch.sigmoid(H_tilde_post)  # (B, n), range [0, 2] as per paper
        H_res = self.sinkhorn(H_tilde_res)        # (B, n, n), doubly stochastic

        return H_pre, H_post, H_res


class MHCBlock(nn.Module):
    """
    DPOT Block with full mHC (Manifold-Constrained Hyper-Connections) mechanism.

    Contains two sub-layers: AFNO2D and MLP, each with independent MHC mappings.

    The mHC update rule for each sub-layer:
        x_stream = H^res · x_stream + H^post^T · Sublayer(Aggregate(H^pre, x_stream))

    Where:
        - Aggregate: Weighted sum of streams using normalized H^pre
        - Sublayer: AFNO2D or MLP (unchanged from original)
        - H^res: Doubly stochastic matrix for residual stream mixing
        - H^post: Distribution weights to write sublayer output back to streams

    Args:
        mixing_type: Type of frequency mixer ('afno')
        width: Channel dimension C (DPOT Tiny: 512)
        n_blocks: Number of blocks in AFNO2D (4)
        mlp_ratio: MLP hidden dimension ratio (1)
        channel_first: Whether input is channel-first (True)
        modes: Number of Fourier modes (32)
        act: Activation function ('gelu')
        h, w: Spatial dimensions for AFNO2D
        n_streams: Number of parallel streams (4)
        sinkhorn_iter: Sinkhorn iterations (20)
        alpha_init: MHC gating scalar initialization (0.01)
        complex_norm: Whether to use ComplexGroupNorm in AFNO2D (False)
        sandwich_norm: Whether to use Sandwich-Norm (False)
        use_layer_scale: Whether to use Layer Scale (False)
        use_far: Whether to use Frequency-Aware Residual in AFNO2D (False)
    """
    def __init__(self, mixing_type='afno', double_skip=False, width=32, n_blocks=4,
                 mlp_ratio=1., channel_first=True, modes=32, drop=0., drop_path=0.,
                 act='gelu', h=14, w=8, n_streams=4, sinkhorn_iter=20, alpha_init=0.01,
                 complex_norm=False, sandwich_norm=False, use_layer_scale=False, use_far=False):
        super().__init__()
        self.n_streams = n_streams
        self.width = width
        self.act_fn = ACTIVATION[act]
        self.sandwich_norm = sandwich_norm
        self.use_layer_scale = use_layer_scale

        # Original components (unchanged)
        self.norm1 = nn.GroupNorm(8, width)
        if mixing_type == "afno":
            self.filter = AFNO2D(width=width, num_blocks=n_blocks, sparsity_threshold=0.01,
                                channel_first=channel_first, modes=modes,
                                hard_thresholding_fraction=1, hidden_size_factor=1, act=act,
                                complex_norm=complex_norm, use_far=use_far)

        self.norm2 = nn.GroupNorm(8, width)

        # Sandwich-Norm: additional post-normalization layers
        if sandwich_norm:
            self.norm1_post = nn.GroupNorm(8, width)
            self.norm2_post = nn.GroupNorm(8, width)

        # Layer Scale: learnable scaling factors for residual branches (from CaiT paper)
        # Initialized to small values (0.1) to let network start close to identity
        if use_layer_scale:
            self.gamma1 = nn.Parameter(torch.ones(width) * 0.01)
            self.gamma2 = nn.Parameter(torch.ones(width) * 0.01)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act_fn,
            nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        # MHC mappings for AFNO sub-layer
        self.mhc_afno = MHCMapping(n_streams, width, sinkhorn_iter, alpha_init)

        # MHC mappings for MLP sub-layer
        self.mhc_mlp = MHCMapping(n_streams, width, sinkhorn_iter, alpha_init)

    def forward(self, x_stream):
        """
        Args:
            x_stream: (B, n, C, H, W) multi-stream hidden state

        Returns:
            (B, n, C, H, W) updated multi-stream hidden state
        """
        B, n, C, H, W = x_stream.shape

        # ========== AFNO Sub-layer ==========
        # 1. Compute mapping matrices
        H_pre, H_post, H_res = self.mhc_afno(x_stream)

        # 2. Aggregate: H^pre · x → (B, C, H, W)
        # Normalize H_pre so it sums to 1 (proper weighted average)
        H_pre_norm = H_pre / (H_pre.sum(dim=-1, keepdim=True) + 1e-8)  # (B, n)
        sub_in = torch.einsum('bn, bnchw -> bchw', H_pre_norm, x_stream)  # (B, C, H, W)

        # 3. Sub-layer processing (Norm → AFNO2D → optional post-Norm)
        sub_in = self.norm1(sub_in)
        sub_out = self.filter(sub_in)  # AFNO2D has internal residual
        if self.sandwich_norm:
            sub_out = self.norm1_post(sub_out)

        # Apply Layer Scale if enabled
        if self.use_layer_scale:
            sub_out = self.gamma1.view(1, -1, 1, 1) * sub_out

        # 4. Distribute: (H^post)^T · sub_out → (B, n, C, H, W)
        # H_post: (B, n), sub_out: (B, C, H, W)
        sub_out_expand = H_post.view(B, n, 1, 1, 1) * sub_out.unsqueeze(1)  # (B, n, C, H, W)

        # 5. Residual mixing: H^res · x_stream
        # H_res: (B, n, n), x_stream: (B, n, C, H, W)
        x_mixed = torch.einsum('bmn, bnchw -> bmchw', H_res, x_stream)  # (B, n, C, H, W)

        # 6. Update: x_stream = H^res · x_stream + H^post^T · sub_out
        x_stream = x_mixed + sub_out_expand

        # ========== MLP Sub-layer ==========
        # 1. Compute mapping matrices
        H_pre, H_post, H_res = self.mhc_mlp(x_stream)

        # 2. Aggregate
        H_pre_norm = H_pre / (H_pre.sum(dim=-1, keepdim=True) + 1e-8)
        sub_in = torch.einsum('bn, bnchw -> bchw', H_pre_norm, x_stream)

        # 3. Sub-layer processing (Norm → MLP → optional post-Norm)
        sub_in = self.norm2(sub_in)
        sub_out = self.mlp(sub_in)
        if self.sandwich_norm:
            sub_out = self.norm2_post(sub_out)

        # Apply Layer Scale if enabled
        if self.use_layer_scale:
            sub_out = self.gamma2.view(1, -1, 1, 1) * sub_out

        # 4. Distribute
        sub_out_expand = H_post.view(B, n, 1, 1, 1) * sub_out.unsqueeze(1)

        # 5. Residual mixing
        x_mixed = torch.einsum('bmn, bnchw -> bmchw', H_res, x_stream)

        # 6. Update
        x_stream = x_mixed + sub_out_expand

        return x_stream


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    use_far: whether to use Frequency-Aware Residual (different residual weights for different frequencies)
    """
    def __init__(self, width = 32, num_blocks=8, channel_first = False,sparsity_threshold=0.01, modes = 32,hard_thresholding_fraction=1, hidden_size_factor=1, act='gelu', complex_norm=False, use_far=False):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"



        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.complex_norm = complex_norm
        self.use_far = use_far
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

        # ComplexGroupNorm between the two MLP layers in frequency domain
        if self.complex_norm:
            # o1 has shape (..., num_blocks, block_size * hidden_size_factor)
            hidden_dim = self.num_blocks * self.block_size * self.hidden_size_factor
            num_groups = min(8, hidden_dim)  # Use 8 groups or less if hidden_dim is small
            self.freq_norm = ComplexGroupNorm(num_groups, hidden_dim)

        # Frequency-Aware Residual: learnable weights for different frequency modes
        # Low frequency gets higher weight (preserve large-scale structure)
        # High frequency gets lower weight (allow more processing/modification)
        if self.use_far:
            self.far_weight = nn.Parameter(self._init_far_weights(modes))

    def _init_far_weights(self, modes):
        """
        Initialize FAR (Frequency-Aware Residual) weights.
        Low frequency modes get higher weight (closer to 1.0) to preserve large-scale structure.
        High frequency modes get lower weight (closer to 0.5) to allow more modification.

        Returns:
            Tensor of shape (modes, modes) with values in range [0.5, 1.0]
        """
        y, x = torch.meshgrid(torch.arange(modes), torch.arange(modes), indexing='ij')
        radius = torch.sqrt(x.float()**2 + y.float()**2)
        # Normalize radius to [0, 1]
        max_radius = radius.max()
        if max_radius > 0:
            radius_norm = radius / max_radius
        else:
            radius_norm = radius
        # Weight: 1.0 at center (low freq), 0.5 at corner (high freq)
        # This means low frequency residual is stronger, high frequency is weaker
        weights = 1.0 - 0.5 * radius_norm
        return weights

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # x = torch.fft.rfft2(x, dim=(1, 2))

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # Apply ComplexGroupNorm between the two MLP layers if enabled
        if self.complex_norm:
            o1_real, o1_imag = self.freq_norm(o1_real, o1_imag)

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # Apply Frequency-Aware Residual (FAR) weights if enabled
        # This scales frequency components: low freq preserved more, high freq modified more
        if self.use_far:
            # far_weight shape: (modes, modes), need to broadcast to (B, modes, modes, num_blocks, block_size)
            far_w = self.far_weight.view(1, kept_modes, kept_modes, 1, 1)
            o2_real[:, :kept_modes, :kept_modes] = o2_real[:, :kept_modes, :kept_modes] * far_w
            o2_imag[:, :kept_modes, :kept_modes] = o2_imag[:, :kept_modes, :kept_modes] * far_w

        x = torch.stack([o2_real, o2_imag], dim=-1)
        ## for ab study
        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")



        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x




_logger = logging.getLogger(__name__)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATION[act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, mixing_type = 'afno', double_skip = True, width = 32, n_blocks = 4, mlp_ratio=1., channel_first = True, modes = 32, drop=0., drop_path=0., act='gelu', h=14, w=8, complex_norm=False, sandwich_norm=False, use_layer_scale=False, use_far=False):
        super().__init__()
        # self.norm1 = norm_layer(width)
        # self.norm1 = torch.nn.LayerNorm([width])
        self.norm1 = torch.nn.GroupNorm(8, width)
        # self.norm1 = torch.nn.InstanceNorm2d(width,affine=True,track_running_stats=False)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]
        self.sandwich_norm = sandwich_norm
        self.use_layer_scale = use_layer_scale

        if mixing_type == "afno":
            self.filter = AFNO2D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first = channel_first, modes = modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act, complex_norm=complex_norm, use_far=use_far)

        self.norm2 = torch.nn.GroupNorm(8, width)

        # Sandwich-Norm: additional post-normalization layers
        if sandwich_norm:
            self.norm1_post = torch.nn.GroupNorm(8, width)
            self.norm2_post = torch.nn.GroupNorm(8, width)

        # Layer Scale: learnable scaling factors for residual branches (from CaiT paper)
        # Initialized to small values (0.1) to let network start close to identity
        if use_layer_scale:
            self.gamma1 = nn.Parameter(torch.ones(width) * 0.01)
            self.gamma2 = nn.Parameter(torch.ones(width) * 0.01)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.sandwich_norm:
            x = self.norm1_post(x)

        # Apply Layer Scale if enabled
        if self.use_layer_scale:
            x = self.gamma1.view(1, -1, 1, 1) * x

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        if self.sandwich_norm:
            x = self.norm2_post(x)

        # Apply Layer Scale if enabled
        if self.use_layer_scale:
            x = self.gamma2.view(1, -1, 1, 1) * x

        x = x + residual

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128,act='gelu'):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C
    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x










class PointwiseTransform(nn.Module):
    """Pointwise linear transform applied to the channel dimension.
    Operates on (..., C) tensors. Initialized as identity so the model
    starts from the same point as vanilla DPOT.
    """
    def __init__(self, dim, hidden_dim=None, act='gelu'):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Initialize as identity mapping
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class DPOTNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type='afno', in_channels=1, out_channels=4,
                 in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, out_layer_dim=32, depth=12, modes=32,
                 mlp_ratio=1., n_cls=12, normalize=False, act='gelu', time_agg='exp_mlp',
                 use_mhc=False, n_streams=4, sinkhorn_iter=20, alpha_init=0.01,
                 complex_norm=False, sandwich_norm=False, use_layer_scale=False, use_far=False,
                 add_mlp=False, mlp_hidden=16):
        '''
        DPOT Network with optional mHC (Manifold-Constrained Hyper-Connections) support.

        :param img_size: input resolution
        :param patch_size: patch size
        :param mixing_type: type of the mixer
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param in_timesteps: number of input timesteps
        :param out_timesteps: number of output timesteps
        :param n_blocks: number of heads/blocks
        :param embed_dim: latent embedding dimension
        :param out_layer_dim: dimension of output convolutional layer
        :param depth: number of layers
        :param modes: number of Fourier modes
        :param mlp_ratio: ratio of MLP dim
        :param n_cls: number of datasets (no influence)
        :param normalize: whether normalize data
        :param act: activation type
        :param time_agg: type of temporal agg layer
        :param use_mhc: whether to use mHC (Manifold-Constrained Hyper-Connections)
        :param n_streams: number of parallel streams for mHC (default: 4)
        :param sinkhorn_iter: number of Sinkhorn iterations for mHC (default: 20)
        :param alpha_init: initial value for mHC gating scalars (default: 0.01)
        :param complex_norm: whether to use ComplexGroupNorm in AFNO2D (default: False)
        :param sandwich_norm: whether to use Sandwich-Norm in Block/MHCBlock (default: False)
        :param use_layer_scale: whether to use Layer Scale (default: False)
        :param use_far: whether to use Frequency-Aware Residual in AFNO2D (default: False)
        '''
        super(DPOTNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channels + 3, embed_dim=out_channels * patch_size + 3, out_dim=embed_dim, act=act)
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls

        # mHC parameters
        self.use_mhc = use_mhc
        self.n_streams = n_streams

        h = img_size // patch_size
        w = h // 2 + 1

        # Choose block type based on use_mhc
        if use_mhc:
            self.blocks = nn.ModuleList([
                MHCBlock(mixing_type=mixing_type, modes=modes,
                         width=embed_dim, mlp_ratio=mlp_ratio, channel_first=True, n_blocks=n_blocks,
                         double_skip=False, h=h, w=w, act=act,
                         n_streams=n_streams, sinkhorn_iter=sinkhorn_iter, alpha_init=alpha_init,
                         complex_norm=complex_norm, sandwich_norm=sandwich_norm,
                         use_layer_scale=use_layer_scale, use_far=use_far)
                for _ in range(depth)])
            # Final aggregation weight for mHC (softmax over n_streams)
            self.final_agg = nn.Parameter(torch.zeros(n_streams))
        else:
            self.blocks = nn.ModuleList([
                Block(mixing_type=mixing_type, modes=modes,
                      width=embed_dim, mlp_ratio=mlp_ratio, channel_first=True, n_blocks=n_blocks,
                      double_skip=False, h=h, w=w, act=act,
                      complex_norm=complex_norm, sandwich_norm=sandwich_norm,
                      use_layer_scale=use_layer_scale, use_far=use_far)
                for _ in range(depth)])


        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * in_channels, embed_dim)


        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)


        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )



        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = mixing_type

        # Pointwise MLP transform at input/output
        self.add_mlp = add_mlp
        if add_mlp:
            self.input_mlp = PointwiseTransform(in_channels, mlp_hidden, act=act)
            self.output_mlp = PointwiseTransform(out_channels, mlp_hidden, act=act)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
            # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid


    ### in/out: B, X, Y, T, C
    def forward(self, x):
        B, _, _, T, _ = x.shape

        # Pointwise input transform
        if self.add_mlp:
            x = self.input_mlp(x)  # (B, X, Y, T, C) -> (B, X, Y, T, C)

        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   #-> B, C, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 3, 1, 2)


        grid = self.get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y c -> b c x y')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        # Process through blocks (with or without mHC)
        if self.use_mhc:
            # ===== mHC: Initialize multi-stream =====
            # (B, C, H, W) → (B, n, C, H, W)
            # Use expand + contiguous to create n copies of the feature
            x_stream = x.unsqueeze(1).expand(-1, self.n_streams, -1, -1, -1).contiguous()

            # ===== Process through MHC blocks =====
            for blk in self.blocks:
                x_stream = blk(x_stream)

            # ===== mHC: Final aggregation =====
            # (B, n, C, H, W) → (B, C, H, W)
            w_final = torch.softmax(self.final_agg, dim=0)  # (n,)
            x = torch.einsum('n, bnchw -> bchw', w_final, x_stream)
        else:
            # Standard processing without mHC
            for blk in self.blocks:
                x = blk(x)



        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma  + mu

        # Pointwise output transform
        if self.add_mlp:
            x = self.output_mlp(x)  # (B, X, Y, T_out, C_out) -> (B, X, Y, T_out, C_out)

        return x, cls_pred

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr



def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


if __name__ == "__main__":
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(4, 20, 20, 6, 3)
    net = DPOTNet(img_size=20, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32,normalize=True)
    y,_ = net(x)
    print(y.shape)