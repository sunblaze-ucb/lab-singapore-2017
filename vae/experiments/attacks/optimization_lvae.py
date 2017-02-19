from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .optimization import Attack as OptimizationAttack


class Attack(OptimizationAttack):
    """Optimization-based targeted attack directly on latent space."""
    name = 'optimization-lvae'

    targeted = True

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        super(Attack, cls).add_options(parser)

        parser.add_argument('--attack-target', type=int, default=9,
                            help='Target class for targeted attack')
        parser.add_argument('--attack-mode', type=str, choices=['mean', 'random', 'index'], default='mean')
        parser.add_argument('--attack-no-reconstruct-target', action='store_true')
        parser.add_argument('--attack-index', type=int, default=0)

    def _build_attack(self, x, y, x_target):
        # Set of generated adversarial examples.
        x_star_var = tf.get_variable('x_star', [
            self.classifier.batch_size,
            self.model.width * self.model.height * self.model.channels
        ], tf.float32, tf.constant_initializer(0))

        # Ensure that x_star is between 0 and 1.
        x_star = (tf.nn.tanh(x_star_var) + 1.0) / 2.0

        # Loss function.
        p_lambda = self.options.optimization_lambda

        # Distance between original and generated adversarial example.
        distance = tf.nn.l2_loss(x - x_star)

        # Assume the same target class for all images.
        y = tf.ones([self.classifier.batch_size], dtype=tf.int64) * self.options.attack_target

        z_x_mean, z_x_log_sigma_sq = self.model.encode_op(x_star, with_variance=True)
        x_hat = self.model.reconstruct_op(x_star)

        # Transform back to logits as TF expects logits not probabilities.
        epsilon = 10e-8
        x_hat = tf.clip_by_value(x_hat, epsilon, 1 - epsilon)
        x_hat = tf.log(x_hat / (1 - x_hat))

        # Reconstruction loss.
        R_loss = self.model.width * self.model.height * self.model.channels * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(slim.flatten(x_hat), slim.flatten(x_target)),
            reduction_indices=-1
        )

        # We clip gradients of KL divergence to prevent NANs.
        KL_loss = tf.reduce_sum(
            -0.5 * tf.reduce_sum(
                1 + tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) -
                tf.square(tf.clip_by_value(z_x_mean, -10.0, 10.0)) -
                tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0)),
                1
            )
        )

        c_loss = KL_loss + R_loss
        loss = p_lambda * distance + c_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.options.optimization_lr
        ).minimize(loss, var_list=[x_star_var])

        return x_star, y, loss, optimizer

    def _get_target_shape(self):
        return self.model.output_width * self.model.output_height * self.model.channels

    def _get_target(self, model, x):
        """Return a target latent vector, depending on the attack mode."""
        dataset = self.model.dataset.get_data().train
        labels = dataset.labels
        shape = [x.shape[0], self._get_target_shape()]

        target = dataset.images[labels == self.options.attack_target]
        source = target
        if not self.options.attack_no_reconstruct_target:
            target = model.reconstruct(target)
        else:
            target = model.input_for_output(target)

        if self.options.attack_mode == 'mean':
            target = np.broadcast_to(np.mean(target, axis=0), shape)
        elif self.options.attack_mode == 'random':
            index = random.randint(0, source.shape[0] - 1)
            target = np.broadcast_to(target[index], shape)
            source = source[index].reshape([1, -1])
        elif self.options.attack_mode == 'index':
            target = np.broadcast_to(target[self.options.attack_index], shape)
            source = source[self.options.attack_index].reshape([1, -1])

        return target, source

    def _get_attack_inputs(self):
        """Get attack input arguments."""
        inputs = super(Attack, self)._get_attack_inputs()
        inputs['x_target'] = tf.placeholder(tf.float32, [self.model.batch_size, self._get_target_shape()])
        return inputs

    def _get_attack_feed(self, inputs, x, y, offset, limit):
        """Get feed dictionary for each attack batch."""
        feed = super(Attack, self)._get_attack_feed(inputs, x, y, offset, limit)
        feed[inputs['x_target']] = self._target_decoding[offset:limit, :]
        return feed

    def adversarial_examples(self, x, y, intensity=0.1):
        self._target_decoding, self._target_source = self._get_target(self.model, x)
        return super(Attack, self).adversarial_examples(x, y)

    def get_target_examples(self):
        """Return target examples that were used when generating adversarial examples."""
        return self._target_decoding, self._target_source
