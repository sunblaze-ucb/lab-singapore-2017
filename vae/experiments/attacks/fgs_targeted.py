import tensorflow as tf

from .. import attack


class Attack(attack.AttackBase):
    """Targeted fast gradient sign attack."""
    name = 'fgs-targeted'

    targeted = True

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        super(Attack, cls).add_options(parser)

        parser.add_argument('--attack-target', type=int, default=9,
                            help='Target class for targeted attack')
        parser.add_argument('--attack-random-target', action='store_true',
                            help='Use a random target for each image')

    def adversarial_noise_op(self, x, y):
        if self.options.attack_random_target:
            # Random target for each image.
            y = tf.random_uniform([self.classifier.batch_size],
                                  minval=0, maxval=self.classifier.num_classes,
                                  dtype=tf.int64)
        else:
            # Assume the same target class for all images.
            y = tf.ones([self.classifier.batch_size], dtype=tf.int64) * self.options.attack_target

        gradients = tf.gradients(self.classifier.loss_op(x, y), x)[0]
        return -tf.sign(gradients), y
