import tensorflow as tf
import tensorflow_addons as tfa

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math
L2_WEIGHT_DECAY = 1e-4


class SegmentationHead(tfkl.Layer):
    def __init__(self, name="seg_head", filters=9, kernel_size=1, upsampling_factor=16, ** kwargs):
        super(SegmentationHead, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        self.conv = tfkl.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer=tfk.regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer=tfk.initializers.LecunNormal())
        self.upsampling = tfkl.UpSampling2D(
            size=self.upsampling_factor, interpolation="bilinear")

    def call(self, inputs):
        x = self.conv(inputs)
        if self.upsampling_factor > 1:
            x = self.upsampling(x)
        return x


class Conv2DReLu(tfkl.Layer):
    def __init__(self, filters, kernel_size, padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.conv = tfkl.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            padding=self.padding, use_bias=False, kernel_regularizer=tfk.regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer="lecun_normal")

        self.bn = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


class DecoderBlock(tfkl.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.conv2 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.upsampling = tfkl.UpSampling2D(
            size=2, interpolation="bilinear")

    def call(self, inputs, skip=None):
        x = self.upsampling(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(tfkl.Layer):
    def __init__(self, decoder_channels, n_skip=3, **kwargs):
        super().__init__(**kwargs)
        self.decoder_channels = decoder_channels
        self.n_skip = n_skip

    def build(self, input_shape):
        self.conv_more = Conv2DReLu(filters=512, kernel_size=3)
        self.blocks = [DecoderBlock(filters=out_ch)
                       for out_ch in self.decoder_channels]

    def call(self, hidden_states, features):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
