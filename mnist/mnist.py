from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

import tensorflow as tf

FLAGS = None


def model(images, is_training=True):
    """A simple MNIST classification model."""
    images = tf.reshape(images, [-1, 28, 28, 1])

    # First convolutional layer with max pooling and ReLU activation.
    conv1 = slim.conv2d(images, 32, [5, 5], activation_fn=tf.nn.relu, scope='conv1')
    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

    # Second convolutional layer with max pooling and ReLU activation.
    conv2 = slim.conv2d(pool1, 64, [5, 5], activation_fn=tf.nn.relu, scope='conv2')
    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

    # First fully connected layer with ReLU activation.
    flat = slim.flatten(pool2)
    fc1 = slim.fully_connected(flat, 1024, activation_fn=tf.nn.relu, scope='fc1')

    # Dropout.
    drop = slim.dropout(fc1, 0.5, is_training=is_training)

    # Fully connected output layer (logits).
    fc2 = slim.fully_connected(drop, 10, activation_fn=None, scope='fc2')
    return fc2


def batch_inputs(images, labels):
    input_images = tf.constant(images)
    input_labels = tf.constant(labels)

    image, label = tf.train.slice_input_producer(
        [input_images, input_labels],
        num_epochs=FLAGS.num_epochs
    )
    label = tf.cast(label, tf.int32)
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size
    )

    return images, labels


def get_inputs(mode, raw=False):
    # Import MNIST data.
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    images = getattr(mnist, mode).images
    labels = getattr(mnist, mode).labels
    if raw:
        return images, labels

    return batch_inputs(images, labels)


def train():
    """Train the model."""
    x, y_true = get_inputs('train')

    # Construct the model.
    y_pred = model(x, is_training=True)

    # Loss function (cross-entropy).
    loss = slim.losses.softmax_cross_entropy(y_pred, y_true)
    tf.summary.scalar('training/loss', loss)

    # Optimizer and training operation.
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Accuracy.
    accuracy = slim.metrics.accuracy(tf.argmax(y_pred, 1),
                                     tf.argmax(y_true, 1))
    tf.summary.scalar('training/accuracy', accuracy)

    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        number_of_steps=5000,
        save_summaries_secs=15,
        save_interval_secs=300
    )


def evaluate(tag, x, y_true):
    """Evaluate trained model accuracy."""
    y_pred = model(x, is_training=False)

    accuracy = slim.metrics.accuracy(tf.argmax(y_pred, 1),
                                     tf.argmax(y_true, 1))
    tf.summary.scalar('{}/accuracy'.format(tag), accuracy)

    slim.evaluation.evaluate_once(
        '',
        os.path.join(FLAGS.log_dir, FLAGS.checkpoint),
        FLAGS.log_dir
    )


def attack_fgsm(x, y_true, y_pred, images, labels):
    images, labels = get_inputs('test', raw=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.int32, [None, 10])
    y_pred = model(x, is_training=False)

    # Restore model from checkpoint.
    restorer = tf.train.Saver()
    with tf.Session() as session:
        restorer.restore(session,
                         os.path.join(FLAGS.log_dir, FLAGS.checkpoint))

        # Loss function for the model.
        loss = slim.losses.softmax_cross_entropy(y_pred, y_true)
        # Compute gradients of the loss function along x.
        gradients = tf.gradients(loss, x)[0]
        # Perturbations are just the sign of the gradients, multiplied
        # by the wanted intensity value.
        perturbations = 0.1 * tf.sign(gradients)

        tf.summary.image('adversary/fgsm/perturbations',
                         tf.reshape(perturbations[:10], [-1, 28, 28, 1]),
                         max_outputs=10)

        examples = x + perturbations
        tf.summary.image('adversary/fgsm/examples',
                         tf.reshape(examples[:10], [-1, 28, 28, 1]),
                         max_outputs=10)

        # Fetch some examples and record summaries.
        summary_op = tf.summary.merge_all()
        examples, summaries = session.run([examples, summary_op], feed_dict={
            x: images[:1000],
            y_true: labels[:1000],
        })

        # Store summaries.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        summary_writer.add_summary(summaries, 5000)
        summary_writer.flush()

    # Evaluate trained model accuracy on adversarial examples.
    tf.reset_default_graph()
    examples, y_true = batch_inputs(examples, labels[:1000])
    evaluate('adversary/fgs', examples, y_true)


def attack_optimization():
    images, labels = get_inputs('test', raw=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.int32, [None, 10])

    scope = tf.get_variable_scope()
    with tf.variable_scope('attack'):
        # Set of generated adversarial examples.
        x_star_var = tf.get_variable('x_star', [FLAGS.batch_size, 784],
                                     tf.float32,
                                     tf.constant_initializer(0))

        # Ensure that x_star is between 0 and 1.
        x_star = (tf.nn.tanh(x_star_var) + 1.0) / 2.0

        # Distance between original and generated adversarial example.
        distance = tf.reduce_mean(tf.nn.l2_loss(x - x_star))

        # Loss function for the model.
        with tf.variable_scope(scope):
            y_pred = model(x_star, is_training=False)

        y = tf.one_hot(y_true, 10)
        y_pred = tf.nn.softmax(y_pred)
        y = tf.cast(tf.reshape(y_true, [-1, 10, 1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1, 1, 10])
        c_loss = tf.reduce_mean(tf.log(1 - tf.batch_matmul(y, y_pred)))

        loss = 0.00000001 * distance - c_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.1
        ).minimize(loss, var_list=[x_star_var])

    # Store summaries of adversarial examaples.
    tf.summary.image('adversary/optimization/examples',
                     tf.reshape(x_star[:10], [-1, 28, 28, 1]),
                     max_outputs=10)

    summary_op = tf.summary.merge_all()

    # Restore model from checkpoint.
    model_vars = [
        var
        for var in slim.get_variables()
        if not var.name.startswith('attack/')
    ]
    restorer = tf.train.Saver(model_vars)
    with tf.Session() as session:
        restorer.restore(session,
                         os.path.join(FLAGS.log_dir, FLAGS.checkpoint))

        # Run optimization.
        session = tf.get_default_session()
        session.run(tf.variables_initializer(slim.get_variables('attack')))

        for i in xrange(200):
            _, examples, c_loss_value, loss_value, l2_value = session.run(
                [optimizer, x_star, c_loss, loss, distance],
                feed_dict={
                    x: images[:100],
                    y_true: labels[:100],
                }
            )

            print('Generating attack, iteration {}: c_loss={} loss={} l2={}'.format(
                i, c_loss_value, loss_value, l2_value
            ))

        # Store summaries.
        summaries = session.run(summary_op)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        summary_writer.add_summary(summaries, 5000)
        summary_writer.flush()

    # Evaluate trained model accuracy on adversarial examples.
    tf.reset_default_graph()
    examples, y_true = batch_inputs(examples, labels[:100])
    evaluate('adversary/optimization', examples, y_true)


def main(_):
    """Entry point."""
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'evaluate':
        x, y_true = get_inputs('test')
        evaluate('test', x, y_true)
    elif FLAGS.mode == 'attack_fgsm':
        attack_fgsm()
    elif FLAGS.mode == 'attack_optimization':
        attack_optimization()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'evaluate', 'attack_fgsm', 'attack_optimization'])
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size.')
    parser.add_argument('--checkpoint', type=str, default='model.ckpt-5000')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='Directory for storing model checkpoints')
    parser.add_argument('--data_dir', type=str, default='.mnist_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
