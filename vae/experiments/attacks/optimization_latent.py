from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf

from .optimization import Attack as OptimizationAttack


class Attack(OptimizationAttack):
    """Optimization-based targeted attack directly on latent space."""
    name = 'optimization-latent'

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        super(Attack, cls).add_options(parser)

        parser.add_argument('--attack-target', type=int, default=9,
                            help='Target class for targeted attack')
        parser.add_argument('--attack-mode', type=str, choices=['mean', 'random', 'index', 'untargeted'],
                            default='mean')
        parser.add_argument('--attack-index', type=int, default=0)

    @property
    def targeted(self):
        return self.options.attack_mode != 'untargeted'

    def _build_attack(self, x, y, z_target):
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
        if self.options.attack_mode != 'untargeted':
            y = tf.ones([self.classifier.batch_size], dtype=tf.int64) * self.options.attack_target
        else:
            y = -y

        if hasattr(self.classifier, 'classifiers'):
            # Special handling for classifier ensemble.
            encodings = []
            weight = 1.0 / len(self.classifier.classifiers)
            for index, classifier in enumerate(self.classifier.classifiers):
                encodings.append(weight * tf.nn.l2_loss(z_target[:, index, :] - classifier.model.encode_op(x_star)))
            c_loss = tf.add_n(encodings)
        else:
            encoding = self.model.encode_op(x_star)
            c_loss = tf.nn.l2_loss(z_target - encoding)

        # For the untargeted case, we invert the sign.
        if self.options.attack_mode == 'untargeted':
            c_loss = -c_loss

        loss = p_lambda * distance + c_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.options.optimization_lr
        ).minimize(loss, var_list=[x_star_var])

        return x_star, y, loss, optimizer

    def _get_target(self, model, x):
        """Return a target latent vector, depending on the attack mode."""
        dataset = self.model.dataset.get_data().train
        labels = dataset.labels

        if self.options.attack_mode in ('mean', 'random', 'index'):
            source = dataset.images[labels == self.options.attack_target]
            latent = model.encode(source)
            if self.options.attack_mode == 'mean':
                latent = np.mean(latent, axis=0)
            elif self.options.attack_mode == 'random':
                index = random.randint(0, source.shape[0] - 1)
                latent = latent[index].reshape([1, -1])
                source = source[index].reshape([1, -1])
            elif self.options.attack_mode == 'index':
                latent = latent[self.options.attack_index].reshape([1, -1])
                source = source[self.options.attack_index].reshape([1, -1])

            latent = np.broadcast_to(latent, [x.shape[0], latent.shape[1]])
        elif self.options.attack_mode == 'untargeted':
            source = None
            latent = model.encode(x)

        return latent, source

    def _get_attack_inputs(self):
        """Get attack input arguments."""
        inputs = super(Attack, self)._get_attack_inputs()
        if hasattr(self.classifier, 'classifiers'):
            inputs['z_target'] = tf.placeholder(tf.float32, [
                self.model.batch_size, len(self.classifier.classifiers), self.model.latent_dim])
        else:
            inputs['z_target'] = tf.placeholder(tf.float32, [self.model.batch_size, self.model.latent_dim])
        return inputs

    def _get_attack_feed(self, inputs, x, y, offset, limit):
        """Get feed dictionary for each attack batch."""
        feed = super(Attack, self)._get_attack_feed(inputs, x, y, offset, limit)
        if hasattr(self.classifier, 'classifiers'):
            feed[inputs['z_target']] = self._target_encoding[offset:limit, :, :]
        else:
            feed[inputs['z_target']] = self._target_encoding[offset:limit, :]
        return feed

    def adversarial_examples(self, x, y, intensity=0.1):
        if hasattr(self.classifier, 'classifiers'):
            self._target_source = []
            self._target_encoding = np.zeros([x.shape[0], len(self.classifier.classifiers), self.model.latent_dim])
            for index, classifier in enumerate(self.classifier.classifiers):
                latent, source = self._get_target(classifier.model, x)
                self._target_source.append(source)
                self._target_encoding[:, index, :] = latent
        else:
            self._target_encoding, self._target_source = self._get_target(self.model, x)

        return super(Attack, self).adversarial_examples(x, y)

    def get_target_examples(self):
        """Return target examples that were used when generating adversarial examples."""
        return self.model.decode(self._target_encoding), self._target_source
