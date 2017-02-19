"""General framework for defining models used in experiments."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tqdm

from . import utils


class ModelBase(object):
    """Abstract model."""

    name = None
    batch_size = 32
    examples_per_epoch = 60000

    def __init__(self, session, dataset, batch_size=None):
        assert self.name, 'Each model must define a name attribute.'

        self.session = session
        self.dataset = dataset
        if batch_size is not None:
            self.batch_size = batch_size

        self._training = tf.placeholder(tf.bool, [], name='training')
        self._model = tf.make_template('model-{}'.format(self.name), self._build)
        self._train = tf.make_template('train-{}'.format(self.name), self._build_train)

    def get_model_filename(self):
        return 'models/{}-{}.weights.tfmod'.format(self.dataset.name, self.name)

    def save(self, filename=None):
        """Saves the model to a file."""
        if not filename:
            filename = self.get_model_filename()
        saver = tf.train.Saver(self._get_model_variables())
        saver.save(self.session, filename)

    def save_graph(self, directory):
        writer = tf.train.SummaryWriter(directory, graph=self.session.graph)
        writer.close()

    def load(self, filename=None):
        """Loads the model from a file."""
        if not filename:
            filename = self.get_model_filename()

        # Transform variable names when loading.
        if self._model.var_scope.name != self._model._name:
            variable_map = {}
            for variable in self._get_model_variables():
                name = variable.name.replace(self._model.var_scope.name, self._model._name)[:-2]
                variable_map[name] = variable

            saver = tf.train.Saver(variable_map)
        else:
            saver = tf.train.Saver(self._get_model_variables())

        tf.train.Saver.restore(saver, self.session, filename)

    def _set_training(self, feed_dict, training):
        """Set any required training placeholders."""
        feed_dict[self._training] = training
        return feed_dict

    def _get_model_variables(self, scope=None, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """Return all model variables."""
        model_scope = self._model.var_scope.name
        if scope is not None:
            model_scope = '{}/{}'.format(model_scope, scope)
        else:
            model_scope = '{}/'.format(model_scope)
        return slim.get_variables(model_scope, collection=collection)

    def _get_train_variables(self, scope=None):
        """Return all training variables."""
        train_scope = self._train.var_scope.name
        if scope is not None:
            train_scope = '{}/{}'.format(train_scope, scope)
        else:
            train_scope = '{}/'.format(train_scope)
        return slim.get_variables(train_scope)

    def _get_num_gpus(self):
        """Return the number of available GPUs."""
        return max(1, len(utils.get_available_gpus()))

    def _get_input_placeholder(self, batches=1):
        raise NotImplementedError

    def _get_latent_placeholder(self, batches=1):
        raise NotImplementedError

    def _get_labels_placeholder(self, batches=1):
        raise NotImplementedError

    def _build(self, x):
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def _build_loss(self, model, labels):
        raise NotImplementedError

    def _build_gradients(self, optimizer, gradients, loss):
        raise NotImplementedError

    def _build_apply_gradients(self, optimizer, gradients, global_step):
        raise NotImplementedError

    def _build_train(self):
        # Detect the number of GPUs.
        num_gpus = self._get_num_gpus()

        # Create a variable to count number of train calls.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # Build optimizer.
        learning_rate, optimizer = self._build_optimizer()

        # Define the network for each GPU.
        tower_gradients = {}
        tower_losses = {}
        tower_lrs = []
        inputs = self._get_input_placeholder(num_gpus)
        labels = self._get_labels_placeholder(num_gpus)
        for gpu in xrange(num_gpus):
            with tf.device('/gpu:{}'.format(gpu)):
                with tf.name_scope('tower_{}'.format(gpu)) as scope:
                    # Grab this portion of the input.
                    input_slice = inputs[gpu * self.batch_size:(gpu + 1) * self.batch_size, :]
                    label_slice = None
                    if labels is not None:
                        label_slice = labels[gpu * self.batch_size:(gpu + 1) * self.batch_size]

                    # Instantiate the model on this tower.
                    model = self._model(input_slice)

                    # Ensure that all batch normalization update operations happen before loss calculations.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                    with tf.control_dependencies(update_ops):
                        # Calculate the loss for this tower.
                        loss = self._build_loss(model, label_slice)

                    self._build_gradients(optimizer, tower_gradients, loss)
                    if not isinstance(loss, (list, tuple)):
                        loss = (loss,)

                    for index, loss_item in enumerate(loss):
                        tower_losses.setdefault(index, []).append([loss_item])

                    lr_adjustments = self._get_learning_rate_adjustments(model)
                    if lr_adjustments is not None:
                        tower_lrs.append(lr_adjustments)

        # Average the gradients.
        average_gradients = {}
        for key, gradients in tower_gradients.items():
            average_gradients[key] = utils.average_gradients(gradients)

        # Average the losses.
        average_loss = []
        for index, loss in sorted(tower_losses.items(), key=lambda x: x[0]):
            loss = tf.concat(0, loss)
            average_loss.append(tf.reduce_mean(loss))

        # Average the learning rate adjustments.
        average_lr_adjustments = []
        if tower_lrs:
            for index in range(len(tower_lrs[0])):
                average_lr_adjustments.append(tf.reduce_mean(tf.concat(0, [x[index] for x in tower_lrs]), 0))

        # Apply the gradients with our optimizers.
        return (
            learning_rate,
            self._build_apply_gradients(optimizer, average_gradients, global_step),
            inputs,
            labels,
            average_loss,
            average_lr_adjustments,
        )

    def _initialize_learning_rate_adjustments(self):
        """Return initial learning rate adjustment parameters."""
        pass

    def _get_learning_rate_adjustments(self, model):
        """Return learning rate adjustment tensors."""
        pass

    def _adjust_learning_rate(self, adjustments, learning_rate, feed_dict):
        """Perform any learning rate adjustments."""
        pass

    def build(self):
        """Build the model."""
        dummy_placeholder = self._get_input_placeholder()
        self._model(dummy_placeholder)

    def train(self, data_sets, epochs, checkpoint=None, checkpoint_every=1):
        """Train the model."""
        if not checkpoint:
            checkpoint = self.get_model_filename()

        # Detect the number of GPUs.
        num_gpus = self._get_num_gpus()
        # Build training operation.
        learning_rate, train_op, inputs, labels, loss_op, lr_adjust_op = self._train()
        # Initialize learning rate adjustments.
        learning_rate_adjustments = self._initialize_learning_rate_adjustments()
        # Initialize variables.
        variables = self._get_model_variables() + self._get_train_variables()
        self.session.run(tf.variables_initializer(variables))
        # How many batches are in an epoch.
        total_batch = int(np.floor(self.examples_per_epoch / (self.batch_size * num_gpus)))

        saver = tf.train.Saver(variables)
        for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
            progress = tqdm.tqdm(total=total_batch, desc='Batch (loss=?.???)', leave=False)
            for i in range(total_batch):
                batch_input, batch_labels = data_sets.train.next_batch(self.batch_size * num_gpus)

                feed_dict = self._set_training({inputs: batch_input}, True)

                if labels is not None:
                    feed_dict[labels] = batch_labels

                op_dict = {
                    'train': train_op,
                    'loss': loss_op,
                    'lr_adjust': lr_adjust_op,
                }

                self._adjust_learning_rate(learning_rate_adjustments, learning_rate, feed_dict)
                result = self.session.run(op_dict, feed_dict=feed_dict)

                # Perform learning rate adjustment if defined.
                learning_rate_adjustments = result['lr_adjust']

                # Show losses.
                losses = ', '.join([
                    'loss{0}={1:.3f}'.format(index, loss)
                    for index, loss in enumerate(result['loss'])
                ])
                progress.set_description('Batch ({})'.format(losses))
                progress.update()
            progress.close()

            # Checkpoint model.
            if epoch % checkpoint_every == 0:
                saver.save(self.session, checkpoint)

        saver.save(self.session, checkpoint)

    def batch_apply(self, operation, feed_dict):
        """Runs on operation by splitting inputs into batches."""
        batches = 1
        for value in feed_dict.values():
            if not np.isscalar(value):
                batches = max(1, len(value) / self.batch_size)

        # Apply operation in batches.
        output = {}
        for batch in xrange(batches):
            batch_feed_dict = {}
            for input_tensor, value in feed_dict.items():
                if np.isscalar(value):
                    batch_feed_dict[input_tensor] = value
                else:
                    batch_feed_dict[input_tensor] = value[batch * self.batch_size:(batch + 1) * self.batch_size]

            result = self.session.run(operation, feed_dict=batch_feed_dict)
            if isinstance(result, tuple):
                for index, value in enumerate(result):
                    output.setdefault(index, []).append(value)
            else:
                output.setdefault(0, []).append(result)

        results = []
        for key in sorted(output.keys()):
            try:
                result = np.concatenate(output[key])
            except ValueError:
                # Properly handle the case where result of each operation is a scalar.
                result = np.asarray(output[key])

            results.append(result)

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def _cache_op(self, name, op, placeholder='input', **parameters):
        if not hasattr(self, '_cached_ops'):
            self._cached_ops = {}

        key = (name, tuple(sorted(parameters.items())))
        if key not in self._cached_ops:
            op_input = getattr(self, '_get_{}_placeholder'.format(placeholder))()
            self._cached_ops[key] = op_input, op(op_input, **parameters)
        return self._cached_ops[key]


class GenerativeModelBase(ModelBase):
    """Abstract model."""

    name = None

    def __init__(self, session, dataset, batch_size=32, latent_dim=50):
        super(GenerativeModelBase, self).__init__(session, dataset, batch_size)

        self.width = dataset.width
        self.height = dataset.height
        self.channels = dataset.channels
        self.latent_dim = latent_dim
        self.defaults = {}

    @property
    def output_dimensions(self):
        return self.width, self.height

    @property
    def output_width(self):
        return self.output_dimensions[0]

    @property
    def output_height(self):
        return self.output_dimensions[1]

    def input_for_output(self, x):
        if self.output_dimensions == (self.width, self.height):
            return x

        def pad_or_crop(x, dim, amount):
            amount_before = int(np.ceil(np.abs(amount) / 2.0))
            amount_after = int(np.floor(np.abs(amount) / 2.0))

            if amount > 0:
                # Pad.
                operation = [(0, 0)] * 4
                operation[dim] = (amount_before, amount_after)
                x = np.pad(x, operation, 'constant')
            else:
                # Crop.
                operation = [slice(None)] * 4
                operation[dim] = slice(amount_before, -amount_after)
                x = x[tuple(operation)]

            return x

        x = x.reshape([-1, self.width, self.height, self.channels])
        x = pad_or_crop(x, 1, self.output_height - self.height)
        x = pad_or_crop(x, 2, self.output_width - self.width)
        x = x.reshape([-1, self.output_width * self.output_height * self.channels])
        return x

    def set_defaults(self, reconstruction=None):
        self.defaults = {
            'reconstruction': reconstruction or {},
        }

    def _get_input_placeholder(self, batches=1):
        return tf.placeholder(tf.float32, [self.batch_size * batches, self.width * self.height * self.channels])

    def _get_latent_placeholder(self, batches=1):
        return tf.placeholder(tf.float32, [self.batch_size * batches, self.latent_dim])

    def _get_labels_placeholder(self, batches=1):
        return None

    def encode_op(self, x, sample=False):
        """Return an operation for encoding an input into a latent representation."""
        raise NotImplementedError

    def encode(self, x, sample=False):
        """Encode an input into a latent representation."""
        encoder_input, encoder_op = self._cache_op('encoder', self.encode_op, sample=sample)
        return self.batch_apply(encoder_op, feed_dict=self._set_training({encoder_input: x}, False))

    def decode_op(self, z):
        """Return an operation for decoding a latent representation."""
        raise NotImplementedError

    def decode(self, z):
        """Decode a latent representation."""
        decoder_input, decoder_op = self._cache_op('decoder', self.decode_op, placeholder='latent')
        return self.batch_apply(decoder_op, feed_dict=self._set_training({decoder_input: z}, False))

    def reconstruct_op(self, x, sample=False, sample_times=1):
        """Return an operation for reconstructing an input using the model."""
        raise NotImplementedError

    def reconstruct(self, x, sample=False, sample_times=None):
        """Reconstruct an input using the model."""
        if sample_times is None:
            sample_times = self.defaults['reconstruction'].get('sampling', 0)
            if sample_times:
                sample = True
                sample_times = sample_times
            else:
                sample = False
                sample_times = 1

        reconstructor_input, reconstructor_op = self._cache_op(
            'reconstructor', self.reconstruct_op, sample=sample, sample_times=sample_times)
        return self.batch_apply(reconstructor_op, feed_dict=self._set_training({reconstructor_input: x}, False))
