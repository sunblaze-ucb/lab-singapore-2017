import tensorflow as tf

from .. import attack


class Attack(attack.AttackBase):
    """Random noise attack."""
    name = 'random'

    def adversarial_noise_op(self, x, y):
        return tf.random_normal(tf.shape(x)), -y
