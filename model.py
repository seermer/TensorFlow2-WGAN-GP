import tensorflow as tf
from tensorflow.keras import layers, Sequential, activations, Model, optimizers
from tensorflow.keras import backend as K


def get_generator(latent_shape):
    model = [
        layers.InputLayer(latent_shape),

        layers.Conv2DTranspose(filters=64, kernel_size=7, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2DTranspose(filters=48, kernel_size=3, padding="same", strides=2, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.UpSampling2D(),

        layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same"),
        layers.Activation(activations.tanh),
    ]
    return Sequential(model)


def get_critic(img_shape):
    model = [
        layers.InputLayer(img_shape),
        layers.Conv2D(filters=24, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=2, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=80, kernel_size=3, strides=2, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=80, kernel_size=3, strides=1, padding="same", use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=96, kernel_size=2, strides=1, use_bias=False),
        layers.LayerNormalization(epsilon=1e-4),
        layers.LeakyReLU(.1),

        layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same", use_bias=False),
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
    def __init__(self,
                 latent_shape=96,
                 img_shape=(28, 28, 1),
                 critic=None,
                 generator=None,
                 critic_step=5,
                 lamb=10.,
                 standardize_img=True):
        super(GAN, self).__init__()
        if (generator is None) and (img_shape != (28, 28, 1)):
            raise ValueError("the default generator generates img of shape (28, 28, 1), "
                             "if you want to output a different image shape, "
                             "please pass in a custom generator")
        if (not isinstance(latent_shape, int)) and (generator is None):
            raise ValueError("latent shape must be a single integer for default generator, "
                             "if you want to input a different shape, "
                             "please pass in a custom generator")
        self.latent_shape = (1, 1, latent_shape)
        self.critic_step = critic_step
        self.lamb = lamb
        self.critic = get_critic(img_shape) if critic is None else critic
        self.generator = get_generator(self.latent_shape) if generator is None else generator
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
        for _ in range(self.critic_step):
            latent = tf.random.uniform(shape=(bs,) + self.latent_shape, minval=-1., maxval=1., dtype=real_img.dtype)
            generated_img = self.generator(latent, training=False)
            with tf.GradientTape(persistent=True) as tape:
                c_loss = (tf.reduce_mean(self.critic(generated_img, training=True)) -
                          tf.reduce_mean(self.critic(real_img, training=True)) +
                          self.lamb * self._gradient_penalty(real_img, generated_img, bs))
            c_grad = tape.gradient(c_loss, self.critic.trainable_weights)
            self.critic_opt.apply_gradients(zip(c_grad, self.critic.trainable_weights))
            retval["c_loss"] = c_loss

        latent = tf.random.uniform(shape=(bs,) + self.latent_shape, minval=-1., maxval=1., dtype=real_img.dtype)
        with tf.GradientTape() as tape:
            g_loss = - tf.reduce_mean(self.critic(self.generator(latent, training=True), training=True))
            g_grad = tape.gradient(g_loss, self.generator.trainable_weights)
            self.generator_opt.apply_gradients(zip(g_grad, self.generator.trainable_weights))
        retval["g_loss"] = g_loss
        return retval
