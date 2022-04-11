import ml_collections

def get_b16_none():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.pretrained_filename = "ViT-B_16.npz"
    config.image_size = 224
    config.patch_size = 16
    config.n_layers = 12
    config.hidden_size = 768
    config.n_heads = 12
    config.name = "B16_None"
    config.mlp_dim = 3072
    config.dropout = 0.1
    config.filters = 9
    config.kernel_size = 1
    config.upsampling_factor = 16
    config.hybrid = False
    return config


def get_b16_cup():
    """Returns the ViT-B/16 configuration."""
    config = get_b16_none()
    config.name = "B16_CUP"
    config.upsampling_factor = 1
    config.decoder_channels = [256, 128, 64, 16]
    config.n_skip = 0
    return config


def get_r50_b16():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_cup()
    config.image_size = 224
    config.name = "R50-B16_CUP"
    config.pretrained_filename = "R50+ViT-B_16.npz"
    config.decoder_channels = [256, 128, 64, 16]
    config.n_skip = 0
    config.hybrid = True
    config.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.n_layers = (3,4,9)
    config.resnet.width_factor = 1
    return config


def get_transunet():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_r50_b16()
    config.name = "TransUNet"
    config.n_skip = 3
    return config
