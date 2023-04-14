from swin_transformer import SwinTransformer


def swin_t(num_classes, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    # Swin Tiny
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_s(num_classes, hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    # Swin Small
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_b(num_classes, hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    # Swin Big
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)


def swin_l(num_classes, hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    # Swin Large
    return SwinTransformer(layers=layers,
                           hidden_dim=hidden_dim,
                           heads=heads,
                           num_classes=num_classes,
                           **kwargs)
