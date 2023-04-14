from torch import nn


class MLP(nn.Module):
    """
    Feed forward MLP layer used in Swin Transformer. Has 2 linear layers with a
    GELU activation function in between. This module maps the input (which) is
    of <dim> dimension to a hidden shape <hidden_dim> and then back to <dim>.
    Input:
        - dim: dimension of the input vector
        - hidden_dim: dimension of the hidden layer
    """

    def __init__(self, dim, hidden_dim, drop_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        # input shape: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.view(b, h, w, c)  # (B, H, W, C)
        x = self.net(x)  # (B, H, W, C)
        x = x.view(b, c, h, w)  # (B, H, W, C)
        return x
