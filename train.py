import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np

import model
import callbacks


def get_data():
    (train_x, _), (test_x, _) = tf.keras.datasets.mnist.load_data()
    AUTOTUNE = tf.data.AUTOTUNE
    del _
    train_x = np.concatenate([train_x, test_x])
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1)).astype(np.float32)
    train_x = train_x / 127.5 - 1.
    train_ds = tf.data.Dataset.from_tensor_slices(train_x)
    train_ds = train_ds.shuffle(len(train_x)).batch(model.BATCH_SIZE).prefetch(AUTOTUNE)
    return train_ds


def train():
    gan = model.GAN(standardize_img=False)
    data = get_data()
    gan.compile()
    monitor = callbacks.GANMonitor()
    gan.fit(x=data,
            batch_size=model.BATCH_SIZE,
            epochs=5,
            callbacks=[monitor])


if __name__ == '__main__':
    train()
