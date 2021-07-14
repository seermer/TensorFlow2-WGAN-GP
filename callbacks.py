import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from model import LATENT_SHAPE

class GANMonitor(Callback):
    def __init__(self, examples=16, num_batches=400):
        super(GANMonitor, self).__init__()
        self.examples = examples
        self.num_batches=num_batches

    def on_batch_end(self, batch, logs=None):
        if batch % self.num_batches != 1:
            return
        latent = tf.random.uniform(shape=(self.examples,) + LATENT_SHAPE, minval=-1., maxval=1., dtype=tf.float32)
        generated_images = self.model.generator(latent, training=False)
        generated_images = (generated_images + 1.) * 127.5
        plt.figure(figsize=(4, 4))

        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis("off")
        plt.show()