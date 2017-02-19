import tensorflow as tf


class AttackBase(object):
    """Abstract attack."""
    name = None

    targeted = False

    def __init__(self, model, classifier, options):
        assert self.name, 'Each attack must define a name attribute.'

        self.session = model.session
        self.model = model
        self.classifier = classifier
        self.options = options

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        pass

    def adversarial_noise_op(self, x, y):
        raise NotImplementedError

    def adversarial_examples(self, x, y, intensity=0.1):
        if hasattr(self, '_adversarial'):
            adversarial_input, adversarial_labels, adversarial_op = self._adversarial
        else:
            with tf.name_scope(self.name):
                adversarial_input = self.classifier._get_input_placeholder()
                adversarial_labels = self.classifier._get_labels_placeholder()
                adversarial_op = self.adversarial_noise_op(adversarial_input, adversarial_labels)
            self._adversarial = adversarial_input, adversarial_labels, adversarial_op

        noise, targets = self.classifier.batch_apply(adversarial_op, feed_dict=self.classifier._set_training({
            adversarial_input: x,
            adversarial_labels: y,
        }, False))

        return (x + intensity * noise, targets)

    def get_target_examples(self):
        """Return target examples that were used when generating adversarial examples."""
        pass
