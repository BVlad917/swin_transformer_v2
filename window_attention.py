import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from cyclic_shift import CyclicShift
from utils import get_log_spaced_relative_distances, create_mask


class WindowAttention(nn.Module):
    """
    Window attention used in the Swin Transformer. Applies the following transformations to an
    input of shape (B, C, H, W) (transformations in order):
        - apply cyclic shift (if necessary)
        - find multi-headed dot product
        - apply normalization by scaling the dot product by the inverse square root of the head dimension
        - apply relative positional embeddings to the scaled dot product
        - apply the attention mask (if necessary)
        - find the values of the multi-headed attention mechanism and concatenate
        - linearly project the values to the input dimension
    """

    def __init__(self, input_dim, heads, head_dim, window_size, shifted, attn_drop_prob, proj_drop_prob):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.shifted = shifted
        self.input_dim = input_dim
        self.tau = nn.Parameter(0.1 * torch.ones(heads))  # tau = scale parameter, non-shared between heads
        inner_dim = heads * head_dim

        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj_drop = nn.Dropout(proj_drop_prob)
        self.pos_emb_drop = nn.Dropout(proj_drop_prob)

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

        self.relative_indices = get_log_spaced_relative_distances(window_size)
        # (WINDOW_SIZE ** 2, WINDOW_SIZE ** 2, 2)

        # continuous relative position bias MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, heads, bias=False)  # bias was set to False here in the original implementation
        )

        if shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.up_to_down_mask = nn.Parameter(create_mask(window_size=window_size,
                                                            displacement=displacement,
                                                            up_to_down=True,
                                                            left_to_right=False),
                                                requires_grad=False)
            self.left_to_right_mask = nn.Parameter(create_mask(window_size=window_size,
                                                               displacement=displacement,
                                                               up_to_down=False,
                                                               left_to_right=True),
                                                   requires_grad=False)

    def forward(self, x):
        # input shape: (B, C, H, W)
        b, c, h, w = x.shape
        if self.shifted:
            x = self.cyclic_shift(x)
        x = x.view(b, h, w, c)  # (B, H, W, C)

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3-tuple of (B, H, W, INNER_DIM)
        nw_h = h // self.window_size  # number of windows along the height
        nw_w = w // self.window_size  # number of windows along the width

        # reshape q, k, v into appropriate representations for cosine attention
        q, k, v = map(lambda t: rearrange(t, "b (nw_h w_h) (nw_w w_w) (heads d) -> b heads (nw_h nw_w) (w_h w_w) d",
                                          w_h=self.window_size, w_w=self.window_size, heads=self.heads,
                                          d=self.head_dim),
                      qkv)  # (B HEADS NUM_WINDOWS^2 WINDOW_SIZE^2 HEAD_DIM) each

        # scaled cosine attention between Q and K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        dots = torch.einsum("b h w i d, b h w j d -> b h w i j", q, k)
        # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, WINDOW_SIZE^2)

        tau = torch.clamp(self.tau, min=0.01)  # tau always >= 0.01
        tau = rearrange(tau, "h -> 1 h 1 1 1")  # change shape to allow for broadcasting
        dots = dots / tau  # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, WINDOW_SIZE^2)

        # add positional embeddings
        rel_pos_embeddings = self.cpb_mlp(self.relative_indices)  # (WINDOW_SIZE ** 2, WINDOW_SIZE ** 2, HEADS)
        rel_pos_embeddings = rearrange(rel_pos_embeddings, "i j h -> 1 h 1 i j")
        # (1, HEADS, 1, WINDOW_SIZE^2, WINDOW_SIZE^2)
        dots = dots + rel_pos_embeddings
        # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, WINDOW_SIZE^2)

        # apply the mask (if shifted), apply softmax, and apply dropout
        if self.shifted:
            dots[:, :, -nw_w:] += self.up_to_down_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_to_right_mask
        attn = dots.softmax(dim=-1)  # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, WINDOW_SIZE^2)
        attn = self.attn_drop(attn)  # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, WINDOW_SIZE^2)

        # find the output of the attention mechanism, apply dropout, and rearrange back to initial shape
        out = torch.einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        # (B, HEADS, NUM_WINDOWS^2, WINDOW_SIZE^2, HEAD_DIM)
        out = rearrange(out, "b heads (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (heads d)",
                        w_h=self.window_size,
                        w_w=self.window_size,
                        heads=self.heads,
                        d=self.head_dim,
                        nw_w=nw_w,
                        nw_h=nw_h)  # (B, H, W, INNER_DIM)
        out = self.to_out(out)  # (B, H, W, C)
        out = self.proj_drop(out)  # (B, H, W, C)

        out = out.view(b, c, h, w)  # (B, C, H, W)
        if self.shifted:
            out = self.cyclic_back_shift(out)  # (B, C, H, W)

        return out
