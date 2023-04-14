from torch import nn

from mlp import MLP
from pre_norm import PreNorm
from residual import Residual
from window_attention import WindowAttention


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block. Contains (in order):
        1. Layer Normalization
        2. Window Attention (shifted or not)
        3. Residual connection
        4. Layer Normalization
        5. Feed forward MLP layer
        6. Residual connection
    """

    def __init__(self, input_dim, heads, head_dim, window_size, shifted, mlp_dim, drop_path_prob, attn_drop_prob,
                 proj_drop_prob):
        super().__init__()
        window_attention = WindowAttention(input_dim=input_dim,
                                           heads=heads,
                                           head_dim=head_dim,
                                           window_size=window_size,
                                           shifted=shifted,
                                           attn_drop_prob=attn_drop_prob,
                                           proj_drop_prob=proj_drop_prob)
        self.norm1 = PreNorm(dim=input_dim, fn=window_attention)
        self.attention_block = Residual(fn=self.norm1, drop_path_prob=drop_path_prob)

        mlp = MLP(dim=input_dim, hidden_dim=mlp_dim, drop_prob=proj_drop_prob)
        self.norm2 = PreNorm(dim=input_dim, fn=mlp)
        self.mlp_block = Residual(fn=self.norm2, drop_path_prob=drop_path_prob)

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.attention_block(x)  # (B, C, H, W)
        x = self.mlp_block(x)  # (B, C, H, W)
        return x
