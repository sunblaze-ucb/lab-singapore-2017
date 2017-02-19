import tensorflow as tf
import tensorflow.contrib.slim as slim

from .. import classifier


class Classifier(classifier.ClassifierBase):
    """Simple classifier with two fully-connected hidden layers."""
    name = 'simple-classifier'

    def _build(self, x):
        # Run the encoder of the underlying model to get the latent representation.
        loops = 1 if not self.sample else 10
        z_x_sampled = []
        for _ in xrange(loops):
            z_x_sampled.append(self.model.encode_op(x, sample=self.sample))

        # Compute the mean sampled value.
        z_x = tf.add_n(z_x_sampled) / len(z_x_sampled)

        # Classify based on latent space.
        with slim.arg_scope([slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': self._training}):
            # Classify based on latent space.
            fc1 = slim.fully_connected(z_x, 512, activation_fn=tf.nn.relu, scope='fc1')
            fc2 = slim.fully_connected(fc1, 512, activation_fn=tf.nn.relu, scope='fc2')
            # Don't use softmax on output due to our loss function.
            return slim.fully_connected(fc2, 10, activation_fn=tf.identity, scope='fc_out')

    def _build_loss(self, model, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(model, labels)
