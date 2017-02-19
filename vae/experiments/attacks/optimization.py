import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .. import attack


class Attack(attack.AttackBase):
    """Optimization-based attack."""
    name = 'optimization'

    def __init__(self, *args, **kwargs):
        super(Attack, self).__init__(*args, **kwargs)

        self._attack = tf.make_template('attack-{}'.format(self.name), self._build_attack)

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        parser.add_argument('--optimization-lambda', type=float, default=1.0)
        parser.add_argument('--optimization-epochs', type=int, default=100)
        parser.add_argument('--optimization-lr', type=float, default=0.001)

    def _build_attack(self, x, y):
        # Set of generated adversarial examples.
        x_star_var = tf.get_variable('x_star', [
            self.classifier.batch_size,
            self.model.width * self.model.height * self.model.channels
        ], tf.float32, tf.constant_initializer(0))

        # Ensure that x_star is between 0 and 1.
        x_star = (tf.nn.tanh(x_star_var) + 1.0) / 2.0

        # Loss function.
        num_classes = self.classifier.num_classes
        p_lambda = self.options.optimization_lambda

        # Distance between original and generated adversarial example.
        distance = tf.nn.l2_loss(x - x_star)

        # Classification error.
        y = tf.one_hot(y, num_classes)
        y_predict = tf.nn.softmax(self.classifier.predict_op(x_star))
        y = tf.reshape(y, [-1, num_classes, 1])
        y_predict = tf.reshape(y_predict, [-1, 1, num_classes])
        c_loss = tf.reduce_mean(tf.log(1 - tf.batch_matmul(y, y_predict)))

        loss = p_lambda * distance - c_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.options.optimization_lr
        ).minimize(loss, var_list=[x_star_var])

        return x_star, y, loss, optimizer

    def _get_attack_inputs(self):
        """Get attack input arguments."""
        return {
            'x': self.classifier._get_input_placeholder(),
            'y': self.classifier._get_labels_placeholder(),
        }

    def _get_attack_feed(self, inputs, x, y, offset, limit):
        """Get feed dictionary for each attack batch."""
        return {
            inputs['x']: x[offset:limit, :],
            inputs['y']: y[offset:limit],
        }

    def adversarial_examples(self, x, y, intensity=0.1):
        inputs = self._get_attack_inputs()
        x_star, target, loss, optimizer = self._attack(**inputs)
        loss = tf.reduce_mean(loss)

        # Do some optimization steps to generate proper output.
        total_batch = int(np.floor(x.shape[0] / self.classifier.batch_size))

        x_star_batches = []
        target_batches = []
        for batch in tqdm.trange(total_batch, desc='Batch'):
            # Initialize optimizer.
            self.session.run(tf.variables_initializer(slim.get_variables(self._attack.var_scope.name)))

            offset = batch * self.classifier.batch_size
            limit = (batch + 1) * self.classifier.batch_size

            progress = tqdm.tqdm(total=self.options.optimization_epochs, desc='Epoch (loss=?.???)', leave=False)
            for epoch in xrange(self.options.optimization_epochs):
                _, x_star_value, target_value, loss_value = self.session.run(
                    [optimizer, x_star, target, loss],
                    feed_dict=self.classifier._set_training(self._get_attack_feed(inputs, x, y, offset, limit), False)
                )

                progress.set_description('Epoch (loss={0:.3f})'.format(loss_value))
                progress.update()
            progress.close()

            x_star_batches.append(x_star_value)
            target_batches.append(target_value)

        return (np.concatenate(x_star_batches), np.concatenate(target_batches))
