import torch
from torch import nn
from timm.models.layers import trunc_normal_

from stage_module import StageModule


class SwinTransformer(nn.Module):
    """
    Swin Transformer architecture. Contains 4 stages followed by an MLP head.
    """

    def __init__(self, layers, hidden_dim, heads, num_classes, channels=3, downscaling_factors=(4, 2, 2, 2),
                 head_dim=32, window_size=7, drop_path_rate=0.1, attn_drop_prob=0., proj_drop_prob=0.,
                 patch_merge_drop_prob=0.):
        super().__init__()
        assert len(layers) == 4, "Swin Transformer has 4 stages by default"
        dpr = torch.linspace(0, drop_path_rate, sum(layers)).tolist()

        self.stage1 = StageModule(num_layers=layers[0],
                                  in_channels=channels,
                                  downscaling_factor=downscaling_factors[0],
                                  hidden_dim=hidden_dim,
                                  num_heads=heads[0],
                                  head_dim=head_dim,
                                  window_size=window_size,
                                  drop_path_probs=dpr[:sum(layers[:1])],
                                  attn_drop_prob=attn_drop_prob,
                                  proj_drop_prob=proj_drop_prob,
                                  patch_merge_drop_prob=patch_merge_drop_prob)
        self.stage2 = StageModule(num_layers=layers[1],
                                  in_channels=hidden_dim,
                                  downscaling_factor=downscaling_factors[1],
                                  hidden_dim=hidden_dim * 2,
                                  num_heads=heads[1],
                                  head_dim=head_dim,
                                  window_size=window_size,
                                  drop_path_probs=dpr[sum(layers[:1]): sum(layers[:2])],
                                  attn_drop_prob=attn_drop_prob,
                                  proj_drop_prob=proj_drop_prob,
                                  patch_merge_drop_prob=patch_merge_drop_prob)
        self.stage3 = StageModule(num_layers=layers[2],
                                  in_channels=hidden_dim * 2,
                                  downscaling_factor=downscaling_factors[2],
                                  hidden_dim=hidden_dim * 4,
                                  num_heads=heads[2],
                                  head_dim=head_dim,
                                  window_size=window_size,
                                  drop_path_probs=dpr[sum(layers[:2]): sum(layers[:3])],
                                  attn_drop_prob=attn_drop_prob,
                                  proj_drop_prob=proj_drop_prob,
                                  patch_merge_drop_prob=patch_merge_drop_prob)
        self.stage4 = StageModule(num_layers=layers[3],
                                  in_channels=hidden_dim * 4,
                                  downscaling_factor=downscaling_factors[3],
                                  hidden_dim=hidden_dim * 8,
                                  num_heads=heads[3],
                                  head_dim=head_dim,
                                  window_size=window_size,
                                  drop_path_probs=dpr[sum(layers[:3]): sum(layers)],
                                  attn_drop_prob=attn_drop_prob,
                                  proj_drop_prob=proj_drop_prob,
                                  patch_merge_drop_prob=patch_merge_drop_prob)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

        # init weights
        self.apply(self._init_weights)
        self.stage1._init_norms()
        self.stage2._init_norms()
        self.stage3._init_norms()
        self.stage4._init_norms()

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = self.stage1(x)  # (B, hidden_dim, H // downscaling_factors[0], W // downscaling_factors[0])
        x = self.stage2(x)  # (B, hidden_dim * 2, H // downscaling_factors[1], W // downscaling_factors[1])
        x = self.stage3(x)  # (B, hidden_dim * 4, H // downscaling_factors[2], W // downscaling_factors[2])
        x = self.stage4(x)  # (B, hidden_dim * 8, H // downscaling_factors[3], W // downscaling_factors[3])
        x = x.mean(dim=(-1, -2))  # mean over height and width => (B, hidden_dim * 8)
        x = self.mlp_head(x)  # (B, num_classes)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
