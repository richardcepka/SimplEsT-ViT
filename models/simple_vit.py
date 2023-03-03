# based on: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import torch
import torch.nn.functional as F
from torch import nn


# Helpers
# _________________________________________________________________
def pair(t): 
    return t if isinstance(t, tuple) else (t, t)
# _________________________________________________________________

# Model
# _________________________________________________________________
def posemb_sincos_2d(dim, h, w, device, dtype, temperature=10000):
    assert (dim % 4) == 0, "Feature dimension must be multiple of 4 for sincos emb."

    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias),
        )
    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0

        self.dim_head = dim // heads
        self.heads = heads
        self.scale = self.dim_head**-0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.flash = True 
        if not self.flash: print("Not Using Flash Attention CUDA Kernels")

    def forward(self, x):
        x = self.norm(x)

        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = F.scaled_dot_product_attention(q, k, v)  
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, N, D)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(self, 
                 image_size, patch_size, num_classes, 
                 dim, mlp_dim, depth, heads, 
                 channels=3, drop_p=0
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Conv2d(
            channels, dim, 
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            bias=True,
        )

        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.drop = nn.Dropout(p=drop_p)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

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