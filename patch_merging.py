from torch import nn


class PatchMerging(nn.Module):
    """
    Patch merging layer used in the Swin Transformer. Takes an input, downscales it by
    a factor if <downscaling_factor> by concatenating the <downscaling_factor> x <downscaling_factor>
    spatial neighbours in the depth dimension, and then applies a linear layer on the depth
    dimension (usually meant to decrease the depth size e.g., halfen it)
    Input:
        - in_channels: depth of the input
        - out_channels: desired depth of the output
        - downscaling_factor: by how much to reduce the spatial dimension
    """

    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.out_channels = out_channels
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        # input shape: (B, C, H, W)
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)  # (B, downscaling_factor^2 * C, H // downscaling_factor * W // downscaling_factor)
        x = x.view(b, new_h, new_w,
                   -1)  # (B, H // downscaling_factor, W // downscaling_factor, downscaling_factor^2 * C)
        x = self.linear(x)  # (B, H // downscaling_factor, W // downscaling_factor, out_channels)
        x = x.view(b, -1, new_h, new_w)  # (B, out_channels, H // downscaling_factor, W // downscaling_factor)
        return x
