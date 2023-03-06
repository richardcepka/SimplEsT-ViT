import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


# DKS/TAT https://github.com/deepmind/dks/blob/main/dks/pytorch/parameter_sampling_functions.py
# _________________________________________________________________
def scaled_uniform_orthogonal_(weights, gain=1.0, delta=True):
    """Initializes fully-connected or conv weights using the SUO distribution.
    Similar to torch.nn.init.orthogonal_, except that it supports Delta
    initializations, and sampled weights are rescaled by
    ``max(sqrt(out_dim / in_dim), 1)``, so that the layer preserves q values at
    initialization-time (assuming initial biases of zero).
    Note that as with all PyTorch functions ending with '_', this function
    modifies the value of its tensor argument in-place.
    Should be used with a zeros initializer for the bias parameters for DKS/TAT.
    See the "Parameter distributions" section of DKS paper
    (https://arxiv.org/abs/2110.01765) for a discussion of the SUO distribution
    and Delta initializations.
    Args:
        weights: A PyTorch Tensor corresponding to the weights to be randomly
        initialized.
        gain: A float giving an additional scale factor applied on top of the
        standard recaling used in the SUO distribution. This should be left
        at its default value when using DKS/TAT. (Default: 1.0)
        delta: A bool determining whether or not to use a Delta initialization
        (which zeros out all weights except those in the central location of
        convolutional filter banks). (Default: True)
    Returns:
        The ``weights`` argument (whose value will be initialized).
    """

    shape = list(weights.size())

    if delta and len(shape) != 2:
        # We assume 'weights' is a filter bank when len(shape) != 2

        # In PyTorch, conv filter banks have that shape
        # [in_dim, out_dim, loc_dim_1, loc_dim_2]
        in_dim = shape[0]
        out_dim = shape[1]

        rescale_factor = max(math.sqrt(out_dim / in_dim), 1.0)

        nonzero_part = torch.nn.init.orthogonal_(weights.new_empty(in_dim, out_dim),
                                                gain=(rescale_factor * gain))

        if any(s % 2 != 1 for s in shape[2:]):
            raise ValueError("All spatial axes must have odd length for Delta "
                            "initializations.")

        midpoints = [(s - 1) // 2 for s in shape[2:]]
        indices = [slice(None), slice(None)] + midpoints

        with torch.no_grad():
            weights.fill_(0.0)
            weights.__setitem__(indices, nonzero_part)

        return weights

    else:

        # torch.nn.orthogonal_ flattens dimensions [1:] instead of [:-1], which is
        # the opposite of what we want here. So we'll first compute the version with
        # the first two dimensions swapped, and then we'll transpose at the end.

        shape = [shape[1], shape[0]] + shape[2:]

        in_dim = math.prod(shape[1:])
        out_dim = shape[0]

        rescale_factor = max(math.sqrt(out_dim / in_dim), 1.0)

        weights_t = torch.nn.init.orthogonal_(weights.new_empty(shape),
                                            gain=(rescale_factor * gain))
        with torch.no_grad():
            return weights.copy_(weights_t.transpose_(0, 1))
# _________________________________________________________________

# TAT https://arxiv.org/pdf/2203.08120.pdf
# _________________________________________________________________
def lrelu_cmap(c, neg_slope):
    # Equation 9.
    return (
        c 
        + 
        (
            ((1 - neg_slope)**2)
            /
            (math.pi * (1 + neg_slope**2))
        ) 
        * 
        (math.sqrt(1 - c**2) - (c * math.acos(c)))
    )


def binary_search(f, target, input_, min_, max_, tol=1e-8, max_eval=100):
    for _ in range(max_eval):
        value = f(input_)
        if math.isinf(value) or math.isnan(value): raise ValueError(f"Function returned {value}.") 

        if abs(value - target) < tol:
            return input_
        
        if value > target:
            max_ = input_

        elif value < target:
            min_ = input_
          
        input_ = 0.5 * (min_ + max_)

    raise ValueError(
        f"Maximum evaluations ({max_eval}) exceeded while searching "
        "for solution. This is probably due the specified target "
        "being unachievable for the given architecture. For example,"
        " a Leaky-ReLU MLP of only a few layers may not be able "
        "to achieve the default C(0) target of 0.9 under TAT. "
        "Intuitively, this is because a shallow network cannot be "
        "made sufficiently nonlinear with such activation functions."
        " The solution to this would be to either use a smaller "
        "value for the C(0) target (corresponding to a more linear "
        "model), or to a use a deeper architecture."
    )


class TReLU(nn.Module):
    def __init__(self, neg_slope):
        super().__init__()
        # Lemma 1.
        self.neg_slope = neg_slope
        self.output_scale = math.sqrt(2.0 / (1.0 + neg_slope**2))

    def forward(self, x):
        return self.output_scale * F.leaky_relu(x, negative_slope=self.neg_slope, inplace=False)
# _________________________________________________________________

# E-SPA https://arxiv.org/pdf/2302.10322.pdf
# _________________________________________________________________
def gamma_to_alpha(gamma):
    return math.sqrt(1 - math.exp(-2 * gamma))


def decompose_attention(A):
    assert (A > 0).sum() == math.prod(A.shape)

    row_sum = A.sum(1)
    B = torch.log(A / row_sum)
    return row_sum, B


def get_decomposed_kernel_matrix(dim, depth, alpha_max_depth, max_depth, inverse):
    if depth == 0:
        return torch.eye(dim)

    gamma_depth = get_gamma_depth(depth, alpha_max_depth, max_depth)
    kernel_matrix = get_kernel_matrix(dim, gamma_depth, dtype=torch.double)  # calculate on double precision, float64
    """
    A COMPATIBILITY WITH NON-CAUSAL ATTENTION: (page 15)
    "For our SPA methods, it is straightforward to extend
    to non-causal attention by changing Ll in Eqs. (8) and (9) from 
    being the Cholesky decompositionof Σl to being the (symmetric) 
    matrix square root of Σl."

    Lemma 1.: (page 31)
    "It is clear that Σ is positive semi definite, 
    as it is the covariance matrix of a stationary 
    Ornstein-Uhlenbeck process, hence a Cholesky factor must exist."
    """
    return sqrtpd(kernel_matrix, inverse).float()  # convert back to float32


def get_gamma_depth(depth, alpha_max_depth, max_depth):
    exponent = 2 * depth / max_depth
    return -(1 / 2) * math.log(1 - alpha_max_depth**exponent)


def get_kernel_matrix(dim, gamma_depth, dtype=torch.float):
    # E-SPA
    def _get_all_subtracted_pairs(x, y):
        return x.unsqueeze(1) - y.unsqueeze(0)

    _arange = torch.arange(dim, dtype=dtype)
    return torch.exp(-gamma_depth * _get_all_subtracted_pairs(_arange, _arange).abs())


def sqrtpd(A, inverse=False):
    """Compute the (inverse) square root of a Symmetric positive definite matrix"""
    eigvals, eigvecs = torch.linalg.eigh(A)
    # Compute (inverse) square root of eigenvalues
    threshold = eigvals.max(-1).values * eigvals.size(-1) * torch.finfo(eigvals.dtype).eps
    sqrt_eigvals = (torch.rsqrt if inverse else torch.sqrt)(torch.clamp(eigvals, min=threshold))
    # Compute matrix (inverse) square root
    return (eigvecs * sqrt_eigvals.unsqueeze(-2)) @ eigvecs.t()
# _________________________________________________________________

# Helpers
# _________________________________________________________________
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def compose(f, n):
    def fn(x):
        for _ in range(n):
            x = f(x)
        return x
    return fn
# _________________________________________________________________

# Model
# _________________________________________________________________
def posemb_sincos_2d(dim, h, w, device, dtype, temperature=10000):
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    assert (dim % 4) == 0, "Feature dimension must be multiple of 4 for sincos emb."
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class Layer(nn.Module):
    def __init__(self, in_features, out_features, act_fn, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.act_fn = act_fn

        scaled_uniform_orthogonal_(self.lin.weight)
        if bias: nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.act_fn(self.lin(x))


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, act_fn, bias=True):
        super().__init__()
        self.layer1 = Layer(dim, mlp_dim, act_fn, bias)
        self.layer2 = Layer(mlp_dim, dim, act_fn, bias)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class Attention(nn.Module):
    def __init__(
            self, dim, seq_lenght, depth, 
            max_depth, alpha_max_depth, heads, 
            qkv_bias=True
    ):
        super().__init__()
        assert dim % heads == 0
        assert depth > 0
        self.dim_head = dim // heads
        self.heads = heads
        self.scale = self.dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # ____________________________ E-SPA ______________________________
        # initialize biase to zero
        if qkv_bias:
            nn.init.zeros_(self.to_qkv.bias)

        # initialize q_i with zeros
        torch.nn.init.zeros_(self.to_qkv.weight[:dim, :])
        # initialize k_i and v_i with N(0, 1/dim)
        torch.nn.init.normal_(self.to_qkv.weight[dim:, :], std=dim**-0.5)

        previous_inv = get_decomposed_kernel_matrix(
            seq_lenght, depth - 1, alpha_max_depth, max_depth, inverse=True
        )
        current = get_decomposed_kernel_matrix(
            seq_lenght, depth, alpha_max_depth, max_depth, inverse=False
        )
        attention = (current @ previous_inv).float()
        # make sure all values are positive
        # reason can be numerical precision (E-SPA is only empirically verified that attention is positive)
        min_value = attention.min()
        if min_value <= 0: 
            print(f"Attention at depth {depth} had negative or zero values {min_value}.")
            attention = torch.clamp(attention, min=min_value.abs() + torch.finfo(attention.dtype).eps)

        d, B = decompose_attention(attention)
        self.register_buffer("d", d.reshape(1, 1, -1, 1))
        self.register_buffer("B", B.unsqueeze(0).unsqueeze(0))
        # _________________________________________________________________

        self.flash = True
        if not self.flash: print("Not Using Flash Attention CUDA Kernels")

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = self.d * F.scaled_dot_product_attention(q, k, v, attn_mask=self.B)  
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale + self.B
            attn = self.d * self.attend(dots)
            out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, N, D)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        mlp_dim,
        act_fn,
        seq_lenght,
        max_depth,
        gamma_max_depth,
        heads,
        att_bias,
        ff_biase
    ):
        super().__init__()
        alpha_max_depth = gamma_to_alpha(gamma_max_depth)
        self.layers = []
        for depth in range(1, max_depth + 1):
            self.layers.extend(
                [
                    Attention(dim, seq_lenght, depth, max_depth, alpha_max_depth, heads, att_bias),
                    FeedForward(dim, mlp_dim, act_fn, ff_biase)
                ]
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class SimplEsTViT(nn.Module):
    def __init__(self, 
                 image_size, patch_size, num_classes, 
                 dim, mlp_dim, depth, heads, 
                 c_val_0_target=0.9, gamma_max_depth=0.005, 
                 channels=3, drop_p=0, att_bias=True, ff_biase=True
    ):
        super().__init__()
        # ______________________________ TAT _______________________________
        # Vanilla network, without skip connections. C ○ C ○ ...
        f_newtork_lrelu_c0 = lambda neg_slope: compose(
            partial(lrelu_cmap, neg_slope=neg_slope), 
            2*depth  # 2 layers per depth, each block (FeedForward) contains 2 TReLU nonlinealiries
        )(0.0)  # C_f(0) = c_val_0_target

        neg_slope = binary_search(
            lambda a: c_val_0_target - f_newtork_lrelu_c0(a), 0,
            input_=0.5, 
            min_=0., 
            max_=1.0
        )

        act_fn = TReLU(neg_slope)
        # _________________________________________________________________

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Conv2d(
            channels, dim, 
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            bias=True,
        )
        # scaled_uniform_orthogonal_(self.to_patch_embedding.weight, delta=False)
        nn.init.normal_(self.to_patch_embedding.weight, mean=0.0, std=patch_dim**-0.5)  # N(0, 1/dim), fan-in
        nn.init.zeros_(self.to_patch_embedding.bias)

        self.transformer = Transformer(
            dim, mlp_dim, act_fn, num_patches, depth, gamma_max_depth, heads, att_bias, ff_biase
        )

        self.drop = nn.Dropout(p=drop_p)
        self.linear_head = nn.Linear(dim, num_classes, bias=True)
        scaled_uniform_orthogonal_(self.linear_head.weight)
        nn.init.zeros_(self.linear_head.bias)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        B, D, H, W = x.shape
        pe = posemb_sincos_2d(D, H, W, x.device, x.dtype)
        # b, dim, h, w -> b, h*w, dim
        x = x.reshape(B, D, H * W).permute(0, 2, 1) + pe

        x = self.transformer(x)
        x = self.drop(x)  # better before pooling https://arxiv.org/pdf/2302.06112.pdf
        x = x.mean(dim=1)
        # x = self.drop(x)
        return self.linear_head(x)
