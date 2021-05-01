import tensorflow as tf
import numpy as np
import models.layers as layers
import models.utils as utils

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

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

BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
WEIGHTS = {"imagenet21k": 21_843, "imagenet21k+imagenet2012": 1_000}
SIZES = {"B_16", "B_32", "L_16", "L_32"}


def preprocess_inputs(X):
    """Preprocess images"""
    return tf.keras.applications.imagenet_utils.preprocess_input(
        X, data_format=None, mode="tf"
    )


def build_model(
    image_size: int,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    dropout=0.1
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
    """
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    y = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=False
    )(x)
    y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    n_patch_sqrt = (image_size//patch_size)
    y = tf.keras.layers.Reshape(target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)
    y = layers.SegmentationHead(**CONFIG_SEG_HEAD)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def load_pretrained(size, weights, model):
    """Load model weights for a known configuration."""
    fname = f"ViT-{size}_{weights}.npz"
    origin = f"{BASE_URL}/{fname}"
    local_filepath = tf.keras.utils.get_file(
        fname, origin, cache_subdir="weights")
    utils.load_weights_numpy(model, local_filepath)


def vit_b16(
    image_size: int = 224,
    pretrained=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B16. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=16,
        image_size=image_size
    )
    if pretrained:
        load_pretrained(
            size="B_16", weights=weights, model=model
        )
    return model


def vit_b32(
    image_size: int = 224,
    pretrained=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B32. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_B,
        name="vit-b32",
        patch_size=32,
        image_size=image_size,
    )
    if pretrained:
        load_pretrained(
            size="B_32", weights=weights, model=model
        )
    return model


def vit_l16(
    image_size: int = 384,
    pretrained=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L16. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_L,
        patch_size=16,
        name="vit-l16",
        image_size=image_size,
    )
    if pretrained:
        load_pretrained(
            size="L_16", weights=weights, model=model
        )
    return model


def vit_l32(
    image_size: int = 384,
    pretrained=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L32. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_L,
        patch_size=32,
        name="vit-l32",
        image_size=image_size,
    )
    if pretrained:
        load_pretrained(
            size="L_32", weights=weights, model=model
        )
    return model




CONFIG_SEG_HEAD = {
    "name": "None",
    "filters": 9,
    "kernel_size": 1,
    "upsampling_factor": 16
}


class TransUnet(tfk.Model):
    def __init__(self, image_size=512, *args, **kwargs):
        super(TransUnet, self).__init__(*args, **kwargs)
        self.encoder = vit_b16(image_size=image_size)
        self.encoder.trainable = False
        self.seg_head = SegmentationHead(**CONFIG_SEG_HEAD)
    
    def call(self, inputs):
        y = self.encoder(inputs)
        B, n_patch, hidden = tf.shape(y)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        y = tf.reshape(y, [B, h, w, hidden])
        logits = self.seg_head(y)
        return logits

@tf.function
def segmentation_loss(y_true, y_pred):
    cross_entropy_loss = tf.losses.categorical_crossentropy(
        y_true=y_true, y_pred=y_pred, from_logits=True)
    dice_loss = gen_dice(y_true, y_pred)
    return 0.5 * cross_entropy_loss + 0.5 * dice_loss

@tf.function
def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(
        y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(
        pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

