import tensorflow as tf
from tensorflow.keras import layers, Sequential, activations, Model, optimizers
from tensorflow.keras import backend as K

BATCH_SIZE = 64
LATENT_SHAPE = (1, 1, 96)
IMG_SHAPE = (28, 28, 1)
CRITIC_STEP = 5
LAMBDA = 10.


def get_generator():
    model = [
        layers.InputLayer(LATENT_SHAPE),

        layers.Conv2DTranspose(filters=64, kernel_size=7, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2DTranspose(filters=48, kernel_size=3, padding="same", strides=2, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2DTranspose(filters=32, kernel_size=3, padding="same", strides=2, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=24, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same"),
        layers.Activation(activations.tanh),
    ]
    return Sequential(model)


def get_critic():
    model = [
        layers.InputLayer(IMG_SHAPE),
        layers.Conv2D(filters=24, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=2, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=64, kernel_size=2, strides=1, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=80, kernel_size=3, strides=2, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.DepthwiseConv2D(kernel_size=3, strides=1, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Flatten(),
        layers.Dense(1)
    ]
    return Sequential(model)


class GAN(Model):
    def __init__(self, critic=None, generator=None, standardize_img=True):
        super(GAN, self).__init__()
        self.critic = get_critic() if critic is None else critic
        self.generator = get_generator() if generator is None else generator
        self.standardize_img = standardize_img

    def compile(self, optimizer=None, **kwargs):
        super(GAN, self).compile()
        if optimizer is None:
            optimizer = (optimizers.Adam(1e-4), optimizers.Adam(1e-4))
        elif not (isinstance(optimizer, tuple) or isinstance(optimizer, list)) or len(optimizer) != 2:
            raise ValueError("two optimizers must be passed to compile critic and generator respectively")
        self.critic_opt, self.generator_opt = optimizer

    def _gradient_penalty(self, real, generated, bs):
        generated = tf.cast(generated, real.dtype)
        epsilon = tf.random.uniform(shape=(bs, 1, 1, 1), minval=0., maxval=1., dtype=real.dtype)
        interpolate = real * epsilon + generated * (1. - epsilon)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            interpolate_pred = self.critic(interpolate, training=True)
        interpolate_grad = tape.gradient(target=interpolate_pred, sources=interpolate)
        penalty = (tf.math.reduce_euclidean_norm(interpolate_grad, axis=(1, 2, 3)) - 1.) ** 2
        return tf.reduce_mean(penalty)

    @tf.function
    def train_step(self, real_img):
        bs = K.shape(real_img)[0]
        if self.standardize_img:
            real_img = real_img / 127.5 - 1.
        retval = {}
        for _ in range(CRITIC_STEP):
            latent = tf.random.uniform(shape=(bs,) + LATENT_SHAPE, minval=-1., maxval=1., dtype=real_img.dtype)
            generated_img = self.generator(latent, training=False)
            with tf.GradientTape(persistent=True) as tape:
                c_loss = (tf.reduce_mean(self.critic(generated_img, training=True)) -
                                 tf.reduce_mean(self.critic(real_img, training=True)) +
                                 LAMBDA * self._gradient_penalty(real_img, generated_img, bs))
            c_grad = tape.gradient(c_loss, self.critic.trainable_weights)
            self.critic_opt.apply_gradients(zip(c_grad, self.critic.trainable_weights))
            retval["c_loss"] = c_loss

        latent = tf.random.uniform(shape=(bs,) + LATENT_SHAPE, minval=-1., maxval=1., dtype=real_img.dtype)
        with tf.GradientTape() as tape:
            g_loss = - tf.reduce_mean(self.critic(self.generator(latent, training=True), training=True))
            g_grad = tape.gradient(g_loss, self.generator.trainable_weights)
            self.generator_opt.apply_gradients(zip(g_grad, self.generator.trainable_weights))
        retval["g_loss"] = g_loss
        return retval
