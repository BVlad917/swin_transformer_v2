from torch import nn


class PostNorm(nn.Module):
    """
    Normalization layer (with Layer Norm) applied after applying a function
    Used in Swin V2.
    Input:
        - dim: LayerNorm will be applied on the last D dimensions of the input, where
        D is the shape of <dim>. If D is an integer, then the normalization will be applied
        over the last dimension of this size
        - fn: the function to apply before the normalization
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # input shape: (B, C, H, W)
        x = self.fn(x, **kwargs)  # (B, C, H, W)
        b, c, h, w = x.shape
        x = x.view(b, h, w, c)  # (B, H, W, C)
        x = self.norm(x)  # (B, H, W, C)
        x = x.view(b, c, h, w)  # (B, C, H, W)
        return x
