from swin_transformer import SwinTransformer


def swin_t(num_classes, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    # SwinV2 Tiny
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_s(num_classes, hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    # SwinV2 Small
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_b(num_classes, hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    # SwinV2 Big
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_l(num_classes, hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    # SwinV2 Large
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_h(num_classes, hidden_dim=352, layers=(2, 2, 18, 2), heads=(11, 22, 44, 88), **kwargs):
    # SwinV2 Huge
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_g(num_classes, hidden_dim=512, layers=(2, 2, 42, 4), heads=(16, 32, 64, 128), **kwargs):
    # SwinV2 Giant
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)
