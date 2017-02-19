import tensorflow as tf

from .. import attack


class Attack(attack.AttackBase):
    """Fast gradient sign attack."""
    name = 'fgs'

    def adversarial_noise_op(self, x, y):
        gradients = tf.gradients(self.classifier.loss_op(x, y), x)[0]
        return tf.sign(gradients), -y
