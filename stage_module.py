from torch import nn

from patch_merging import PatchMerging
from swin_transformer_block import SwinTransformerBlock


class StageModule(nn.Module):
    """
    Defines one of the 4 stages from Swin Transformer. Contains (in order):
        1. Patch Merging (equivalently Patch Embedding where needed)
        2. n pairs of (Swin Transformer Block, Shifted Swin Transformer Block)
    """

    def __init__(self, num_layers, in_channels, downscaling_factor, hidden_dim, num_heads, head_dim, window_size,
                 drop_path_probs, attn_drop_prob, proj_drop_prob, pos_emb_drop_prob):
        super().__init__()
        assert num_layers % 2 == 0, "Stage layers need to be divisible by 2 for regular and shifted blocks"
        if isinstance(drop_path_probs, list):
            assert len(drop_path_probs) == num_layers, "Must give exactly one DropPath rate to each layer"

        self.patch_merging = PatchMerging(in_channels=in_channels,
                                          out_channels=hidden_dim,
                                          downscaling_factor=downscaling_factor)
        self.pos_emb_drop = nn.Dropout(pos_emb_drop_prob)

        self.layers = nn.ModuleList([])
        for idx in range(num_layers // 2):
            non_shifted_drop_prob = drop_path_probs[idx] if isinstance(drop_path_probs, list) else drop_path_probs
            shifted_drop_prob = drop_path_probs[idx + 1] if isinstance(drop_path_probs, list) else drop_path_probs

            self.layers.append(nn.ModuleList([
                SwinTransformerBlock(input_dim=hidden_dim,
                                     heads=num_heads,
                                     head_dim=head_dim,
                                     window_size=window_size,
                                     shifted=False,
                                     mlp_dim=hidden_dim * 4,
                                     drop_path_prob=non_shifted_drop_prob,
                                     attn_drop_prob=attn_drop_prob,
                                     proj_drop_prob=proj_drop_prob),
                SwinTransformerBlock(input_dim=hidden_dim,
                                     heads=num_heads,
                                     head_dim=head_dim,
                                     window_size=window_size,
                                     shifted=True,
                                     mlp_dim=hidden_dim * 4,
                                     drop_path_prob=shifted_drop_prob,
                                     attn_drop_prob=attn_drop_prob,
                                     proj_drop_prob=proj_drop_prob)
            ]))

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.patch_merging(x)  # (B, hidden_dim, H // downscaling_factor, W // downscaling_factor)
        x = self.pos_emb_drop(x)  # (B, hidden_dim, H // downscaling_factor, W // downscaling_factor)

        for regular_block, shifted_block in self.layers:
            x = regular_block(x)  # (B, hidden_dim, H // downscaling_factor, W // downscaling_factor)
            x = shifted_block(x)  # (B, hidden_dim, H // downscaling_factor, W // downscaling_factor)
        return x

    def _init_norms(self):
        all_blocks = [block for layer in self.layers for block in layer]
        for block in all_blocks:
            nn.init.zeros_(block.norm1.norm.bias)
            nn.init.zeros_(block.norm1.norm.weight)
            nn.init.zeros_(block.norm2.norm.bias)
            nn.init.zeros_(block.norm2.norm.weight)
