import models.encoder_layers as encoder_layers
import models.decoder_layers as decoder_layers
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import models.utils as utils
from medpy import metric
import tensorflow as tf
import math
tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math
tfkc = tfk.callbacks

N_CLASSES = 9
MODELS_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/'
TRAINING_SAMPLES = 2211


class TransUnet():
    def __init__(self, config):
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.n_layers = config.n_layers
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.name = config.name
        self.mlp_dim = config.mlp_dim
        self.dropout = config.dropout
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.upsampling_factor = config.upsampling_factor
        self.hybrid = config.hybrid
        self.model = self.build_model()

    def build_model(self):
        # Tranformer Encoder
        assert self.image_size % self.patch_size == 0, "image_size must be a multiple of patch_size"
        x = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3))

        # Embedding
        if self.hybrid:
            grid_size = self.config.grid
            self.patch_size = self.image_size // 16 // grid_size[0]
            if self.patch_size == 0:
                self.patch_size = 1

            resnet50v2, features = self.resnet_embeddings(x)
            y = resnet50v2.get_layer("conv4_block6_preact_relu").output
            x = resnet50v2.input
        else:
            y = x
            features = None

        y = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="embedding",
            trainable=False
        )(y)
        y = tf.keras.layers.Reshape(
            (y.shape[1] * y.shape[2], self.hidden_size))(y)
        y = encoder_layers.AddPositionEmbs(
            name="Transformer/posembed_input")(y)

        # Transformer/Encoder
        for n in range(self.n_layers):
            y, _ = encoder_layers.TransformerBlock(
                n_heads=self.n_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                name=f"Transformer/encoderblock_{n}",
            )(y)
        y = tfkl.LayerNormalization(
            epsilon=1e-6, name="Transformer/encoder_norm"
        )(y)

        n_patch_sqrt = int(math.sqrt(y.shape[1]))

        y = tfkl.Reshape(
            target_shape=[n_patch_sqrt, n_patch_sqrt, self.hidden_size])(y)

        # Decoder CUP
        if "decoder_channels" in self.config:
            y = decoder_layers.DecoderCup(
                decoder_channels=self.config.decoder_channels, n_skip=self.config.n_skip)(y, features)

        # Segmentation Head
        y = decoder_layers.SegmentationHead(
            filters=self.filters, kernel_size=self.kernel_size, upsampling_factor=self.upsampling_factor)(y)

        return tfk.models.Model(inputs=x, outputs=y, name=self.name)

    def load_pretrained(self):
        """Load model weights for a known configuration."""
        origin = MODELS_URL + self.config.pretrained_filename
        fname = self.config.pretrained_filename
        local_filepath = tf.keras.utils.get_file(
            fname, origin, cache_subdir="weights")

        utils.load_weights_numpy(self.model, local_filepath)

    def compile(self, lr=None, epochs=150, batch_size=24, validation_samples=260):
        self.load_pretrained()
        if lr is None:
            steps_per_epoch = (
                TRAINING_SAMPLES-validation_samples) // batch_size
            starter_learning_rate = 0.01
            end_learning_rate = 0
            decay_steps = epochs * steps_per_epoch
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                starter_learning_rate,
                decay_steps,
                end_learning_rate,
                power=0.9)

        optimizer = tfa.optimizers.SGDW(
            weight_decay=1e-4, momentum=.9, learning_rate=lr)

        self.model.compile(optimizer=optimizer, loss=[
                           TransUnet.segmentation_loss])

    def train_validate(self, training_dataset, validation_dataset, save_path, validation_samples=260, epochs=150, batch_size=24, show_history=True):
        checkpoint_filepath = save_path + '/checkpoint/'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        steps_per_epoch = (TRAINING_SAMPLES-validation_samples) // batch_size
        history = self.model.fit(training_dataset, epochs=epochs, batch_size=batch_size, verbose=1,
                                 steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, callbacks=[model_checkpoint_callback])

        self.model.load_weights(checkpoint_filepath)

        saved_model_path = save_path + "/model"
        self.save_model(saved_model_path)

        print(f"Model saved in {saved_model_path}")

        if show_history:
            plt.figure()
            plt.plot(history.history["loss"], label="training loss")
            plt.plot(history.history["val_loss"], label="validation loss")
            plt.legend()
            plt.show()

        return history

    def train(self, training_dataset, save_path, epochs=150, batch_size=24, show_history=True):

        steps_per_epoch = TRAINING_SAMPLES // batch_size
        history = self.model.fit(training_dataset, epochs=epochs, batch_size=batch_size, verbose=1,
                                 steps_per_epoch=steps_per_epoch)

        self.save_model(save_path)
        print(f"Model saved in {save_path}")
        if show_history:
            plt.figure()
            plt.plot(history.history["loss"], label="training loss")
            plt.legend()
            plt.show()

        return history

    @tf.function
    def segmentation_loss(y_true, y_pred):
        cross_entropy_loss = tf.losses.categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=True)
        dice_loss = TransUnet.gen_dice(y_true, y_pred)
        return 0.5 * cross_entropy_loss + 0.5 * dice_loss

    @tf.function
    def gen_dice(y_true, y_pred):
        """both tensors are [b, h, w, classes] and y_pred is in logit form"""
        # [b, h, w, classes]
        pred_tensor = tf.nn.softmax(y_pred)
        loss = 0.0
        for c in range(N_CLASSES):
            loss += TransUnet.dice_per_class(y_true[:, c], pred_tensor[:, c])
        return loss/N_CLASSES

    @tf.function
    def dice_per_class(y_true, y_pred, eps=1e-5):
        intersect = tf.reduce_sum(y_true * y_pred)
        y_sum = tf.reduce_sum(y_true * y_true)
        z_sum = tf.reduce_sum(y_pred * y_pred)
        loss = 1 - (2 * intersect + eps) / (z_sum + y_sum + eps)
        return loss

    def resnet_embeddings(self, x):
        resnet50v2 = tfk.applications.ResNet50V2(
            include_top=False, input_shape=(self.image_size, self.image_size, 3))
        resnet50v2.trainable = False
        _ = resnet50v2(x)
        layers = ["conv3_block4_preact_relu",
                  "conv2_block3_preact_relu",
                  "conv1_conv"]

        features = []
        for l in layers:
            features.append(resnet50v2.get_layer(l).output)
        return resnet50v2, features

    def save_model_tpu(self, saved_model_path):
        save_options = tf.saved_model.SaveOptions(
            experimental_io_device='/job:localhost')
        self.model.save(saved_model_path, options=save_options)

    def save_model(self, saved_model_path):
        self.model.save(saved_model_path)

    def load_model(self, saved_model_path):
        model = tf.keras.models.load_model(
            saved_model_path, compile=False)
        self.model = model
        return model

    def load_model_tpu(self, tpu_strategy, saved_model_path):
        with tpu_strategy.scope():
            load_options = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost')
            # model = tf.keras.models.load_model(saved_model_path, options=load_options, custom_objects={'loss': vit.TransUnet.segmentation_loss})
            model = tf.keras.models.load_model(
                saved_model_path, options=load_options, compile=False)
            self.model = model
            return model
