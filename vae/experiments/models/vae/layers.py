import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def encoder(x, width, height, channels=1, latent_dim=50, is_training=True):
    with tf.variable_scope('encoder'):
        x = tf.reshape(x, [-1, height * width * channels])
        output = slim.stack(x, slim.fully_connected, [512], activation_fn=tf.nn.relu, scope='hidden')
        z_mean = slim.fully_connected(output, latent_dim, scope='z_mean')
        z_log_sigma = slim.fully_connected(output, latent_dim, scope='z_log_sigma')
    return z_mean, z_log_sigma


@add_arg_scope
def decoder(z, width, height, channels=1, latent_dim=50, is_training=True):
    with tf.variable_scope('decoder'):
        output = slim.stack(z, slim.fully_connected, [512], activation_fn=tf.nn.relu, scope='hidden')
        output = slim.fully_connected(output, width * height * channels,
                                      activation_fn=tf.nn.sigmoid, scope='output')
    return output
