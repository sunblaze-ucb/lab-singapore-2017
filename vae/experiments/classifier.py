"""General framework for defining classifiers used in experiments."""
import numpy as np
import tensorflow as tf

from .model import ModelBase


class ClassifierBase(ModelBase):
    """Abstract classifier."""

    name = None

    def __init__(self, model, num_classes, sample=False):
        super(ClassifierBase, self).__init__(model.session, model.batch_size)

        self.model = model
        self.num_classes = num_classes
        self.sample = sample

    def get_model_filename(self):
        return 'models/{}-{}-{}.weights.tfmod'.format(self.model.dataset.name, self.name, self.model.name)

    def _set_training(self, feed_dict, training):
        feed_dict = super(ClassifierBase, self)._set_training(feed_dict, training)
        # The model should always be frozen while the classifier is running.
        feed_dict[self.model._training] = False
        return feed_dict

    def _get_input_placeholder(self, batches=1):
        return self.model._get_input_placeholder(batches=batches)

    def _get_labels_placeholder(self, batches=1):
        return tf.placeholder(tf.int64, [self.batch_size * batches])

    def _build_optimizer(self):
        return None, tf.train.AdamOptimizer(1e-3)

    def _build_gradients(self, optimizer, gradients, loss):
        gradients.setdefault('classifier', []).append(
            optimizer.compute_gradients(loss, var_list=self._get_model_variables())
        )

    def _build_apply_gradients(self, optimizer, gradients, global_step):
        return optimizer.apply_gradients(gradients['classifier'], global_step=global_step)

    def loss_op(self, x, y):
        """Return an operation for computing the loss of the classifier."""
        y_pred = self.predict_op(x)
        return self._build_loss(y_pred, y)

    def predict_op(self, x):
        """Return an operation for predicting the logits of given input."""
        return self._model(x)

    def predict_labels_op(self, x):
        """Return an operation for predicting the labels of given input."""
        y_pred = self.predict_op(x)
        return tf.argmax(y_pred, 1)

    def predict_logits(self, x):
        """Predict the logits of given input."""
        if hasattr(self, '_predit_logits'):
            predict_input, predict_op = self._predit_logits
        else:
            predict_input = self._get_input_placeholder()
            predict_op = self.predict_op(predict_input)
            self._predit_logits = predict_input, predict_op

        return self.batch_apply(predict_op, feed_dict=self._set_training({predict_input: x}, False))

    def predict(self, x):
        """Predict the label of given input."""
        if hasattr(self, '_predict'):
            predict_input, predict_op = self._predict
        else:
            predict_input = self._get_input_placeholder()
            predict_op = self.predict_labels_op(predict_input)
            self._predict = predict_input, predict_op

        return self.batch_apply(predict_op, feed_dict=self._set_training({predict_input: x}, False))

    def evaluate_op(self, x, y):
        """Return an operation for evaluating the accuracy of the classifier."""
        with tf.name_scope('evaluate'):
            y_pred = self.predict_op(x)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        y,
                        tf.argmax(y_pred, 1)
                    ),
                    tf.float32
                )
            )

    def evaluate(self, x, y):
        """Evaluate accuracy of the classifier."""
        if hasattr(self, '_evaluate'):
            evaluate_input, evaluate_labels, evaluate_op = self._evaluate
        else:
            evaluate_input = self._get_input_placeholder()
            evaluate_labels = self._get_labels_placeholder()
            evaluate_op = self.evaluate_op(evaluate_input, evaluate_labels)
            self._evaluate = evaluate_input, evaluate_labels, evaluate_op

        return np.mean(self.batch_apply(evaluate_op, feed_dict=self._set_training({
            evaluate_input: x,
            evaluate_labels: y,
        }, False)))
