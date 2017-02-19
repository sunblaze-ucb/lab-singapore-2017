import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def encoder(x, width, height, channels=1, latent_dim=50, is_training=True):
    with tf.variable_scope('encoder'):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'scale': True},
                            activation_fn=tf.nn.elu):
            x = tf.reshape(x, [-1, height, width, channels])
            output = slim.stack(x, slim.conv2d, [64, 128, 256], kernel_size=5, stride=2, scope='conv')
            output = slim.flatten(output)

        z_mean = slim.fully_connected(output, latent_dim, activation_fn=None, scope='z_mean')
        z_log_sigma_sq = slim.fully_connected(output, latent_dim, activation_fn=None, scope='z_log_sigma_sq')
    return z_mean, z_log_sigma_sq


@add_arg_scope
def decoder(z, width, height, channels=1, latent_dim=50, is_training=True):
    # Compute downsampled dimensions based on input width/height.
    d_width = int(np.ceil(width / 8.0))
    d_height = int(np.ceil(height / 8.0))

    with tf.variable_scope('decoder'):
        output = slim.fully_connected(z, d_width * d_height * 256, activation_fn=tf.nn.elu, scope='fc')

        with slim.arg_scope([slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'scale': True},
                            activation_fn=tf.nn.elu):
            output = tf.reshape(output, [-1, d_width, d_height, 256])
            output = slim.stack(output, slim.conv2d_transpose, [256, 128, 32],
                                kernel_size=5, stride=2, scope='deconv1')
            output = slim.conv2d_transpose(output, channels, 1, activation_fn=tf.nn.sigmoid, scope='deconv2')
            output = slim.flatten(output)
    return output


@add_arg_scope
def discriminator(d_i, width, height, channels=1, latent_dim=50, is_training=True):
    with tf.variable_scope('discriminator'):
        d_i = tf.reshape(d_i, [-1, height, width, channels])

        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'scale': True},
                            activation_fn=tf.nn.elu):
            output = slim.conv2d(d_i, 32, 5, stride=1, scope='conv1')
            output = slim.stack(output, slim.conv2d, [128, 256, 256], kernel_size=5, stride=2, scope='conv2')
            output = slim.flatten(output)

        lth_layer = slim.fully_connected(output, 1024, activation_fn=tf.nn.elu, scope='fc_lth')
        discrimination = slim.fully_connected(lth_layer, 1, activation_fn=tf.nn.sigmoid, scope='fc_discrimination')
    return discrimination, lth_layer


@add_arg_scope
def pad_power2(x, width, height, channels):
    x = tf.reshape(x, [-1, height, width, channels])
    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
    w_nearest, h_nearest = get_dimensions(width, height)
    if width == w_nearest and height == h_nearest:
        return x, width, height

    if height != h_nearest:
        padding[1] = [
            int(np.ceil((h_nearest - height) / 2.0)),
            int(np.floor((h_nearest - height) / 2.0)),
        ]
        height = h_nearest

    if width != w_nearest:
        padding[2] = [
            int(np.ceil((w_nearest - width) / 2.0)),
            int(np.floor((w_nearest - width) / 2.0)),
        ]
        width = w_nearest
    x = tf.pad(x, padding)
    x = tf.reshape(x, [-1, height * width * channels])
    return x, width, height


def get_dimensions(width, height):
    return tuple([
        2 ** int(np.ceil(np.log2(dimension)))
        for dimension in (width, height)
    ])
