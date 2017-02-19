from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import layers
from ... import model, utils


class Model(model.GenerativeModelBase):
    """VAE-GAN model."""
    name = 'vae-gan'

    # Results of internal model operations.
    model_type = namedtuple('VAEGAN', [
        'width',
        'height',
        'z_x_mean',
        'z_x_log_sigma_sq',
        'z_x',
        'x_tilde',
        'x_tilde_mean',
        'l_x_tilde',
        'd_x',
        'l_x',
        'd_x_p',
    ])

    @property
    def output_dimensions(self):
        return layers.get_dimensions(self.width, self.height)

    def _build(self, x, sample=1):
        """Builds the model."""

        # Reshape input as needed.
        x, width, height = layers.pad_power2(x, self.width, self.height, self.channels)
        # Normal distribution for GAN sampling.
        z_p = tf.random_normal([self.batch_size, self.latent_dim], 0, 1)
        # Normal distribution for VAE sampling.
        eps = tf.random_normal([self.batch_size, self.latent_dim], 0, 1)

        with slim.arg_scope([layers.encoder, layers.decoder, layers.discriminator],
                            width=width,
                            height=height,
                            channels=self.channels,
                            latent_dim=self.latent_dim,
                            is_training=self._training):
            # Get latent representation for sampling.
            z_x_mean, z_x_log_sigma_sq = layers.encoder(x)
            # Sample from latent space.
            z_x = []
            for _ in xrange(sample):
                z_x.append(tf.add(z_x_mean, tf.mul(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps)))
            if sample > 1:
                z_x = tf.add_n(z_x) / sample
            else:
                z_x = z_x[0]
            # Generate output.
            x_tilde = layers.decoder(z_x)

            _, l_x_tilde = layers.discriminator(x_tilde)

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # Generate reconstruction.
                x_tilde_mean = layers.decoder(z_x_mean)

                # Generate a new image.
                x_p = layers.decoder(z_p)
                # Run discriminator on original inputs.
                d_x, l_x = layers.discriminator(x)
                # Run discriminator on generated outputs.
                d_x_p, _ = layers.discriminator(x_p)

            return self.model_type(
                width=width,
                height=height,
                z_x_mean=z_x_mean,
                z_x_log_sigma_sq=z_x_log_sigma_sq,
                z_x=z_x,
                x_tilde=x_tilde,
                x_tilde_mean=x_tilde_mean,
                l_x=l_x,
                l_x_tilde=l_x_tilde,
                d_x=d_x,
                d_x_p=d_x_p,
            )

    def _build_optimizer(self):
        """Different optimizers are needed for different learning rates."""
        lr_encoder = tf.placeholder(tf.float32, shape=[])
        lr_decoder = tf.placeholder(tf.float32, shape=[])
        lr_discriminator = tf.placeholder(tf.float32, shape=[])
        return (
            [lr_encoder, lr_decoder, lr_discriminator],
            [
                tf.train.AdamOptimizer(lr_encoder, epsilon=1.0),
                tf.train.AdamOptimizer(lr_decoder, epsilon=1.0),
                tf.train.AdamOptimizer(lr_discriminator, epsilon=1.0)
            ]
        )

    def _build_loss(self, model, labels):
        """Loss functions for KL divergence, Discrim, Generator, Lth Layer Similarity."""

        # We clip gradients of KL divergence to prevent NANs.
        KL_loss = tf.reduce_sum(
            -0.5 * tf.reduce_sum(
                1 + tf.clip_by_value(model.z_x_log_sigma_sq, -10.0, 10.0) -
                tf.square(tf.clip_by_value(model.z_x_mean, -10.0, 10.0)) -
                tf.exp(tf.clip_by_value(model.z_x_log_sigma_sq, -10.0, 10.0)),
                1
            )
        ) / model.width / model.height / self.channels

        # Discriminator loss.
        D_loss = tf.reduce_mean(-1.0 * (tf.log(tf.clip_by_value(model.d_x, 1e-5, 1.0)) +
                                        tf.log(tf.clip_by_value(1.0 - model.d_x_p, 1e-5, 1.0))))

        # Generator loss.
        G_loss = tf.reduce_mean(-1.0 * (tf.log(tf.clip_by_value(model.d_x_p, 1e-5, 1.0))))

        # Lth Layer Loss - the 'learned similarity measure'.
        LL_loss = tf.reduce_sum(tf.square(model.l_x - model.l_x_tilde)) / model.width / model.height / self.channels

        # Calculate the losses specific to encoder, decoder, discriminator.
        L_e = tf.clip_by_value(KL_loss + LL_loss, -100, 100)
        L_g = tf.clip_by_value(LL_loss + G_loss, -100, 100)
        L_d = tf.clip_by_value(D_loss, -100, 100)

        return L_e, L_g, L_d

    def _build_gradients(self, optimizers, gradients, losses):
        for name, optimizer, loss in zip(['encoder', 'decoder', 'discriminator'], optimizers, losses):
            gradients.setdefault(name, []).append(
                optimizer.compute_gradients(
                    loss,
                    var_list=self._get_model_variables(name, tf.GraphKeys.TRAINABLE_VARIABLES)
                )
            )

    def _build_apply_gradients(self, optimizers, gradients, global_step):
        operations = []
        for name, optimizer in zip(['encoder', 'decoder', 'discriminator'], optimizers):
            operations.append(optimizer.apply_gradients(gradients[name], global_step=global_step))

        return tf.group(*operations)

    def _initialize_learning_rate_adjustments(self):
        # We balance the decoder and discriminator learning rate by using a sigmoid function,
        # encouraging the decoder and discriminator to be about equal.
        return 0.5, 0.5

    def _get_learning_rate_adjustments(self, model):
        return model.d_x, model.d_x_p

    def _adjust_learning_rate(self, adjustments, learning_rate, feed_dict):
        e_learning_rate = 1e-3
        g_learning_rate = 1e-3
        d_learning_rate = 1e-3
        d_real, d_fake = adjustments

        feed_dict.update({
            # Encoder.
            learning_rate[0]: e_learning_rate * utils.sigmoid(np.mean(d_real), -0.5, 15),
            # Decoder.
            learning_rate[1]: g_learning_rate * utils.sigmoid(np.mean(d_real), -0.5, 15),
            # Discriminator.
            learning_rate[2]: d_learning_rate * utils.sigmoid(np.mean(d_fake), -0.5, 15),
        })

    def encode_op(self, x, sample=False, with_variance=False):
        model = self._model(x)
        if sample:
            return model.z_x
        elif with_variance:
            return model.z_x_mean, model.z_x_log_sigma_sq
        else:
            return model.z_x_mean

    def decode_op(self, z):
        # Compute output dimensions.
        width, height = layers.get_dimensions(self.width, self.height)

        with tf.variable_scope(self._model.var_scope, reuse=True):
            x_tilde = layers.decoder(z,
                                     width=width,
                                     height=height,
                                     channels=self.channels,
                                     latent_dim=self.latent_dim,
                                     is_training=self._training)
            return x_tilde

    def reconstruct_op(self, x, sample=False, sample_times=1):
        model = self._model(x, sample=sample_times)
        if sample:
            return model.x_tilde
        else:
            return model.x_tilde_mean
