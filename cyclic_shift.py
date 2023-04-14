import torch
from torch import nn


class CyclicShift(nn.Module):
    """
    Applies a cyclic shift on the input image (or a 4-dimensional feature tensor).
    Assumes input is in channels-first format i.e., (B, C, H, W)
    Input:
        - displacement: by how much to shift the input; int
    """

    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        # input shape: (B, C, H, W)
        x = torch.roll(x, shifts=(self.displacement, self.displacement), dims=(-2, -1))
        return x  # (B, C, H, W)
