import tensorflow as tf

from .optimization import Attack as OptimizationAttack


class Attack(OptimizationAttack):
    """Optimization-based targeted attack."""
    name = 'optimization-targeted'

    targeted = True

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        super(Attack, cls).add_options(parser)

        parser.add_argument('--attack-target', type=int, default=9,
                            help='Target class for targeted attack')
        parser.add_argument('--attack-random-target', action='store_true',
                            help='Use a random target for each image')

    def _build_attack(self, x, y):
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

        # Classification error.
        if self.options.attack_random_target:
            # Random target for each image.
            y = tf.random_uniform([self.classifier.batch_size],
                                  minval=0, maxval=self.classifier.num_classes,
                                  dtype=tf.int64)
        else:
            # Assume the same target class for all images.
            y = tf.ones([self.classifier.batch_size], dtype=tf.int64) * self.options.attack_target

        if hasattr(self.classifier, 'classifiers'):
            # Special handling for classifier ensemble.
            y_predicts = []
            weight = 1.0 / len(self.classifier.classifiers)
            for classifier in self.classifier.classifiers:
                y_predicts.append(weight * classifier.predict_op(x_star))
            y_predict = tf.add_n(y_predicts)
            y1 = tf.one_hot(y, self.classifier.num_classes)
            y1 = tf.reshape(y1, [-1, self.classifier.num_classes, 1])
            y_predict = tf.reshape(y_predict, [-1, 1, self.classifier.num_classes])
            c_loss = -tf.log(tf.batch_matmul(y_predict, y1))
        else:
            y_predict = self.classifier.predict_op(x_star)
            c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_predict, y)

        loss = p_lambda * distance + c_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.options.optimization_lr
        ).minimize(loss, var_list=[x_star_var])

        return x_star, y, loss, optimizer
