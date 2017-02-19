"""Utility functions."""
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    """Returns the number of available GPUS."""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def clip_to_batch_size(model, x):
    """Clip number of samples to a multiple of the model's batch size."""
    return x[:(x.shape[0] / model.batch_size) * model.batch_size]


def crop_for_model(model, x):
    """Crop images so that they are suitable for the given model."""
    if len(x.shape) > 1:
        dim = int(np.sqrt(x.shape[1] / model.channels))
    else:
        dim = int(np.sqrt(x.shape[0] / model.channels))

    diff_h = dim - model.height
    diff_w = dim - model.width
    if not diff_h and not diff_w:
        return x

    result = x.reshape([-1, dim, dim, model.channels])[
        :,
        int(np.ceil(diff_h / 2.0)):-int(np.floor(diff_h / 2.0)),
        int(np.ceil(diff_w / 2.0)):-int(np.floor(diff_w / 2.0)),
        :
    ].reshape([-1, model.width * model.height * model.channels])

    if len(x.shape) == 1:
        result = result.flatten()

    return result


def get_matching_rate(y_true, y_pred, y_target):
    """Compute matching rate."""
    incorrect = np.sum(y_pred != y_true)
    matching = np.sum(y_pred == y_target)
    if not incorrect:
        return 0, 0, 0.0

    return incorrect, matching, float(matching) / incorrect


def accuracy_combinations(dataset, y_true, y_predict, y_target=None):
    """Computes accuracies for different source classes."""
    accuracies = []
    for src in xrange(dataset.class_count):
        filter_indices = (y_true == src)
        filtered_y_true = y_true[filter_indices]
        filtered_y_predict = y_predict[filter_indices]

        count = filtered_y_true.shape[0]
        if not count:
            accuracies.append((src, 0, 0, 0.0, 0.0))
            continue

        correct = np.sum(filtered_y_predict == filtered_y_true)
        accuracy = np.mean(filtered_y_predict == filtered_y_true)
        if y_target is not None:
            filtered_y_target = y_target[filter_indices]
            matching_rate = get_matching_rate(filtered_y_true, filtered_y_predict, filtered_y_target)[2]
        else:
            matching_rate = 0.0
        accuracies.append((src, correct, count, accuracy, matching_rate))

    return accuracies


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def sigmoid(x, shift, mult):
    """Using this sigmoid to discourage one network overpowering the other."""
    return 1 / (1 + np.exp(-(x + shift) * mult))


def plot_digits(dataset, name, data, n=3, figure_size=6, labels=None):
    """Plots MNIST digits."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(figure_size, figure_size))
    ax = plt.gca()

    d = int(np.sqrt(data.shape[1] / dataset.channels))
    if dataset.channels == 1:
        figure = np.zeros((d * n, d * n))
    else:
        figure = np.zeros((d * n, d * n, dataset.channels))

    for i in xrange(n):
        for j in xrange(n):
            try:
                image = data[i * n + j]
            except IndexError:
                break

            if dataset.channels == 1:
                image = image.reshape([d, d])
            else:
                image = image.reshape([d, d, dataset.channels])

            figure[i * d:(i + 1) * d,
                   j * d:(j + 1) * d] = image

            if labels is not None:
                correct, predicted = labels
                correct = correct[i * n + j]
                predicted = predicted[i * n + j]
                if correct != predicted:
                    ax.annotate(
                        '{}'.format(predicted),
                        xy=(j * d, i * d + 10),
                        xytext=(j * d, i * d + 10),
                        xycoords='data',
                        textcoords='data',
                        color='red',
                    )
                    ax.annotate(
                        '{}'.format(correct),
                        xy=(j * d, i * d + 20),
                        xytext=(j * d, i * d + 20),
                        xycoords='data',
                        textcoords='data',
                        color='blue',
                    )
                else:
                    ax.annotate(
                        '{}'.format(correct),
                        xy=(j * d, i * d + 10),
                        xytext=(j * d, i * d + 10),
                        xycoords='data',
                        textcoords='data',
                        color='green',
                    )

    plt.imshow(figure, cmap='Greys')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/{}.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.savefig('results/{}.pdf'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_adversarial_digits(name, dataset, model,
                            originals, original_reconstructions,
                            original_reconstructions_sample_1,
                            original_reconstructions_sample_12,
                            original_reconstructions_sample_50,
                            adversarial, adversarial_reconstructions,
                            adversarial_reconstructions_sample_1,
                            adversarial_reconstructions_sample_12,
                            adversarial_reconstructions_sample_50,
                            loop1_reconstructions,
                            gt_labels,
                            predicted_labels,
                            loop1_predicted_labels,
                            loop2_predicted_labels,
                            target_labels=None,
                            show_labels=False,
                            show_loops=False,
                            rows=6, cols=2,
                            indices=None):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    image_size = model.output_width
    border = 14
    column_separator = image_size

    if indices is not None:
        originals = originals[indices]
        original_reconstructions = original_reconstructions[indices]
        original_reconstructions_sample_1 = original_reconstructions_sample_1[indices]
        original_reconstructions_sample_12 = original_reconstructions_sample_12[indices]
        original_reconstructions_sample_50 = original_reconstructions_sample_50[indices]
        adversarial = adversarial[indices]
        adversarial_reconstructions = adversarial_reconstructions[indices]
        adversarial_reconstructions_sample_1 = adversarial_reconstructions_sample_1[indices]
        adversarial_reconstructions_sample_12 = adversarial_reconstructions_sample_12[indices]
        adversarial_reconstructions_sample_50 = adversarial_reconstructions_sample_50[indices]
        loop1_reconstructions = loop1_reconstructions[indices]
        gt_labels = gt_labels[indices]
        predicted_labels = predicted_labels[indices]
        loop1_predicted_labels = loop1_predicted_labels[indices]
        loop2_predicted_labels = loop2_predicted_labels[indices]

        if target_labels is not None:
            target_labels = target_labels[indices]

    def prediction_color(predicted, index):
        if predicted[index] == gt_labels[index]:
            return 'green'
        elif target_labels is None or predicted[index] == target_labels[index]:
            return 'red'
        else:
            return 'magenta'

    def get_shape(width, height):
        if dataset.channels == 1:
            return [width, height]
        else:
            return [width, height, dataset.channels]

    original_shape = get_shape(dataset.width, dataset.height)
    output_shape = get_shape(model.output_width, model.output_height)
    original_offset_x = (output_shape[0] - original_shape[0]) / 2
    original_offset_y = (output_shape[1] - original_shape[1]) / 2

    column_width = ((image_size + border) * 10 + column_separator)
    if show_labels:
        column_width += 3 * (border + 5)
    if show_loops:
        column_width += 1 * (image_size + border)
    row_height = image_size + border
    width = cols * column_width
    height = rows * row_height
    dpi = 100
    plt.figure(figsize=(width / float(dpi), (height + 30) / float(dpi)), dpi=dpi)
    ax = plt.gca()
    for row in xrange(rows):
        for col in xrange(cols):
            index = row * cols + col
            offset_x = col * column_width
            offset_y = row * row_height
            ax.add_artist(AnnotationBbox(
                OffsetImage(originals[index].reshape(original_shape), cmap='Greys'),
                xy=(offset_x + original_offset_x, offset_y + original_offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(original_reconstructions[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(original_reconstructions_sample_1[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(original_reconstructions_sample_12[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(original_reconstructions_sample_50[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(adversarial[index].reshape(original_shape), cmap='Greys'),
                xy=(offset_x + original_offset_x, offset_y + original_offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            if show_labels:
                ax.annotate(
                    '{}'.format(predicted_labels[index]),
                    xy=(offset_x, offset_y + 10),
                    xytext=(offset_x, offset_y + 10),
                    xycoords='figure pixels',
                    textcoords='figure pixels',
                    color=prediction_color(predicted_labels, index),
                )
                offset_x += border + 5
            ax.add_artist(AnnotationBbox(
                OffsetImage(adversarial_reconstructions[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(adversarial_reconstructions_sample_1[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(adversarial_reconstructions_sample_12[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            ax.add_artist(AnnotationBbox(
                OffsetImage(adversarial_reconstructions_sample_50[index].reshape(output_shape), cmap='Greys'),
                xy=(offset_x, offset_y),
                xycoords='figure pixels',
                box_alignment=(0, 0),
                frameon=False
            ))
            offset_x += image_size + border
            if show_labels:
                ax.annotate(
                    '{}'.format(loop1_predicted_labels[index]),
                    xy=(offset_x, offset_y + 10),
                    xytext=(offset_x, offset_y + 10),
                    xycoords='figure pixels',
                    textcoords='figure pixels',
                    color=prediction_color(loop1_predicted_labels, index),
                )
                offset_x += border + 5
            if show_loops:
                ax.add_artist(AnnotationBbox(
                    OffsetImage(loop1_reconstructions[index].reshape(output_shape), cmap='Greys'),
                    xy=(offset_x, offset_y),
                    xycoords='figure pixels',
                    box_alignment=(0, 0),
                    frameon=False
                ))
                offset_x += image_size + border
            if show_labels:
                ax.annotate(
                    '{}'.format(loop2_predicted_labels[index]),
                    xy=(offset_x, offset_y + 10),
                    xytext=(offset_x, offset_y + 10),
                    xycoords='figure pixels',
                    textcoords='figure pixels',
                    color=prediction_color(loop2_predicted_labels, index),
                )

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/{}.png'.format(name), bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.savefig('results/{}.pdf'.format(name), bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()


def plot_latent_space(name, models, count=1000, adversarial=None):
    """Plots a t-SNE visualization of latent space."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Encode count images from the test set for each model.
    adversarial_offset = None
    z_x = []
    reconstructions = []
    for model in models:
        x = model.dataset.get_data().test.images[:count]
        y = model.dataset.get_data().test.labels[:count]
        if adversarial is not None:
            adversarial_offset = x.shape[0]
            x = np.concatenate([x, adversarial])

        z_x.append(model.encode(x))
        reconstructions.append(model.decode(z_x[-1]))

    encodings_per_model = z_x[0].shape[0]
    z_x = np.concatenate(z_x)
    reconstructions = np.concatenate(reconstructions)

    def is_adversarial(index):
        if adversarial is None:
            return False

        index = index % encodings_per_model
        return index >= adversarial_offset

    def get_label(index):
        if len(models) > 1:
            return '{}/{}'.format(y[index % encodings_per_model], index / encodings_per_model)
        else:
            return '{}'.format(y[index % encodings_per_model])

    # Compute t-SNE 2D embedding.
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    e_x = tsne.fit_transform(z_x)

    # Plot the 2D embedding.
    # Based on: http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    x_min, x_max = np.min(e_x, 0), np.max(e_x, 0)
    e_x = (e_x - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(e_x.shape[0]):
        if is_adversarial(i):
            ax.plot(e_x[i, 0], e_x[i, 1], marker='o', color='green', zorder=10)
        else:
            ax.text(e_x[i, 0], e_x[i, 1], get_label(i),
                    color=plt.cm.Set1(y[i % encodings_per_model] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    if models[0].channels == 1:
        image_shape = [models[0].width, models[0].height]
    else:
        image_shape = [models[0].width, models[0].height, models[0].channels]

    shown_images = np.array([[1., 1.]])
    for i in range(e_x.shape[0]):
        if is_adversarial(i):
            image = adversarial[i % encodings_per_model - adversarial_offset]
            image = adversarial[i % encodings_per_model - adversarial_offset]
        else:
            image = x[i % encodings_per_model]
            continue

        dist = np.sum((e_x[i] - shown_images) ** 2, 1)
        if np.min(dist) < 0.016:
            # Don't show points that are too close.
            continue

        shown_images = np.r_[shown_images, [e_x[i]]]
        imagebox = AnnotationBbox(
            OffsetImage(image.reshape(image_shape), cmap=plt.cm.gray_r, zoom=0.5),
            e_x[i]
        )
        imagebox.set_zorder(11)
        ax.add_artist(imagebox)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.savefig('results/{}.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.savefig('results/{}.pdf'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()
