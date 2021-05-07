import ml_collections

# image_size: int,
# patch_size: int,
# num_layers: int,
# hidden_size: int,
# num_heads: int,
# name: str,
# mlp_dim: int,
# dropout: float,

# filters: int,
# kernel_size: int,
# upsampling_factor: int


def get_b16_none():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.vit = "B_16"
    config.image_size = 512
    config.patch_size = 16
    config.n_layers = 12
    config.hidden_size = 768
    config.n_heads = 12
    config.name = "b16_none"
    config.mlp_dim = 3072
    config.dropout = 0.1
    config.filters = 9
    config.kernel_size = 1
    config.upsampling_factor = 16
    config.hybrid = False
    return config


def get_b16_cup():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.vit = "B_16"
    config.image_size = 512
    config.patch_size = 16
    config.n_layers = 12
    config.hidden_size = 768
    config.n_heads = 12
    config.name = "b16_cup"
    config.mlp_dim = 3072
    config.dropout = 0.1
    config.filters = 9
    config.kernel_size = 1
    config.upsampling_factor = 1
    config.decoder_channels = [256, 128, 64, 16]
    config.n_skip = 0
    config.hybrid = False
    return config


def get_b16_hybrid():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.vit = "B_16"
    config.image_size = 224
    config.patch_size = 16
    config.n_layers = 12
    config.hidden_size = 768
    config.n_heads = 12
    config.name = "b16_cup"
    config.mlp_dim = 3072
    config.dropout = 0.1
    config.filters = 9
    config.kernel_size = 1
    config.upsampling_factor = 1
    config.decoder_channels = [256, 128, 64, 16]
    config.n_skip = 0
    config.hybrid = True
    config.grid = (16, 16)
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}
