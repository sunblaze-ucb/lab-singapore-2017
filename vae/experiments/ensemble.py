"""Ensemble classifier."""
import tensorflow as tf

from .classifier import ClassifierBase


class EnsembleClassifier(ClassifierBase):
    name = 'ensemble'

    # Combination methods.
    MEAN = 'mean'
    MAJORITY = 'majority'

    def __init__(self, classifiers, combination=MEAN):
        self.classifiers = classifiers
        self.combination = combination

        # Ensure that all models and classifiers are compatible.
        attributes = {}
        for classifier in classifiers:
            for attribute in ('width', 'height', 'channels', 'batch_size', 'dataset'):
                value = getattr(classifier.model, attribute)
                if attribute not in attributes:
                    attributes[attribute] = value
                elif value != attributes[attribute]:
                    raise ValueError('Incompatible models cannot be used in an ensemble.')

            for attribute in ('num_classes',):
                value = getattr(classifier, attribute)
                if attribute not in attributes:
                    attributes[attribute] = value
                elif value != attributes[attribute]:
                    raise ValueError('Incompatible classifiers cannot be used in an ensemble.')

        super(EnsembleClassifier, self).__init__(classifiers[0].model, classifiers[0].num_classes)

    def _set_training(self, feed_dict, training):
        feed_dict = super(EnsembleClassifier, self)._set_training(feed_dict, training)
        for classifier in self.classifiers:
            feed_dict = classifier._set_training(feed_dict, training)
        return feed_dict

    def _build(self, x):
        # Get predictions from all classifiers
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict_op(x))

        if self.combination == EnsembleClassifier.MEAN:
            # Mean of all logits.
            return tf.add_n(predictions, name='ensemble') / len(self.classifiers)
        elif self.combination == EnsembleClassifier.MAJORITY:
            # Majority of all predictions.
            maximums = [tf.one_hot(tf.argmax(prediction, 1), self.num_classes) for prediction in predictions]
            probs = tf.one_hot(tf.argmax(tf.add_n(maximums, name='ensemble'), 1), self.num_classes)
            epsilon = 10e-8
            probs = tf.clip_by_value(probs, epsilon, 1 - epsilon)
            return tf.log(probs)

    def _build_loss(self, model, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(model, labels)
