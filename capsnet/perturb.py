"""
"""
import os
import numpy as np
import scipy.misc
import tensorflow as tf

from six.moves import range

from mnist import load_mnist


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mnist-root-path', None, '')
tf.app.flags.DEFINE_string('ckpt-path', None, '')
tf.app.flags.DEFINE_string('meta-path', None, '')
tf.app.flags.DEFINE_string('result-path', None, '')


def load_datasets():
    """
    load mnist
    """
    path_root = FLAGS.mnist_root_path

    path_train_eigens = os.path.join(path_root, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_root, 'train-labels-idx1-ubyte.gz')
    path_issue_eigens = os.path.join(path_root, 't10k-images-idx3-ubyte.gz')
    path_issue_labels = os.path.join(path_root, 't10k-labels-idx1-ubyte.gz')

    datasets = load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)

    all_eigens = np.concatenate(
        [datasets['train_eigens'], datasets['issue_eigens']], axis=0)
    all_labels = np.concatenate(
        [datasets['train_labels'], datasets['issue_labels']], axis=0)

    eigens = np.zeros_like(all_eigens[:10])
    labels = np.zeros_like(all_labels[:10])

    for i in range(10):
        i_labels = np.where(all_labels[:, i] == 1.0)[0]

        m = np.random.randint(i_labels.size)

        n = i_labels[m]

        eigens[i] = all_eigens[n]
        labels[i] = all_labels[n]

    return eigens, labels


def main(_):
    """
    """
    eigens, labels = load_datasets()

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(FLAGS.meta_path)

        saver.restore(session, FLAGS.ckpt_path)

        graph = tf.get_default_graph()

        images_tensor = graph.get_tensor_by_name('images:0')
        labels_tensor = graph.get_tensor_by_name('labels:0')
        digit_capsules_tensor = graph.get_tensor_by_name('digit_capsules:0')
        inserted_digit_capsules_tensor = \
            graph.get_tensor_by_name('inserted_digit_capsules:0')
        reconstruction_tensor = \
            graph.get_tensor_by_name('reconstructions_from_latent:0')

        # NOTE: fetch digit capsules of all digits
        feeds = {
            images_tensor: eigens,
            labels_tensor: labels,
        }

        digit_capsules = session.run(digit_capsules_tensor, feed_dict=feeds)

        # prepare masks
        masks = np.zeros((11 * 16, 10, 16))

        for j in range(16):
            for i in range(11):
                masks[j * 11 + i, :, j] = 0.05 * float(i) - 0.25

        # pertub all 10 digits
        images = []

        for i in range(10):
            capsule = digit_capsules[i:i+1]
            label = labels[i:i+1]

            feeds = {}

            feeds[inserted_digit_capsules_tensor] = \
                np.tile(capsule, (11 * 16, 1, 1)) + masks
            feeds[labels_tensor] = \
                np.tile(label, (11 * 16, 1))

            reconstructions = \
                session.run(reconstruction_tensor, feed_dict=feeds)

            images.append(reconstructions)

        images = np.concatenate(images, axis=0)
        images = np.reshape(images, (-1, 28))

        images = np.split(images, 1760, axis=0)

        images, temp = [], images

        for i in range(0, 1760, 11):
            images.append(np.concatenate(temp[i:i+11], axis=1))

        images, temp = [], images

        for i in range(0, 160, 16):
            images.append(np.concatenate(temp[i:i+16], axis=0))

        images, temp = [], images

        for i in range(0, 10, 5):
            images.append(np.concatenate(temp[i:i+5], axis=1))

        images = np.concatenate(images, axis=0)

        images = images * 255.0

        images = np.clip(images, 0.0, 255.0).astype(np.uint8)

        scipy.misc.imsave('./zooo.png', images)


if __name__ == '__main__':
    tf.app.run()
