import torch

from swin_variants import swin_t

swin_tiny = swin_t(num_classes=3)
x = torch.randn(10, 3, 224, 224)
y = swin_tiny(x)
print(y.shape)
