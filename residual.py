from torch import nn
from timm.models.layers import DropPath


class Residual(nn.Module):
    """
    Add an input to the output of a function as a residual connection. Also applies stochastic depth.
    Input:
        - fn: the function to apply to the input before adding the residual connection.
        e.g., a PyTorch module
        - drop_path_prob: probability of dropping a layer through stochastic depth; int
    """

    def __init__(self, fn, drop_path_prob):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x, **kwargs):
        # input shape: (B, C, H, W)
        y = self.fn(x, **kwargs)  # assume fn keeps shape (B, C, H, W)
        return self.drop_path(y) + x  # (B, C, H, W)
