"""
"""
import numpy as np
import os
import tensorflow as tf

from six.moves import range
from mnist import load_mnist

from capsnet import build_capsnet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mnist-root-path', None, '')

tf.app.flags.DEFINE_boolean(
    'reconstruction-loss', False, '')

tf.app.flags.DEFINE_integer(
    'batch-size', 128, '')
tf.app.flags.DEFINE_integer(
    'routing-frequency', 3, '')


def load_datasets():
    """
    load mnist
    """
    path_root = FLAGS.mnist_root_path

    path_train_eigens = os.path.join(path_root, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_root, 'train-labels-idx1-ubyte.gz')
    path_issue_eigens = os.path.join(path_root, 't10k-images-idx3-ubyte.gz')
    path_issue_labels = os.path.join(path_root, 't10k-labels-idx1-ubyte.gz')

    return load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)


def mnist_batches(eigens, labels, batch_size, epochs=1000000):
    """
    batch data generator
    """
    step = 0

    indices = np.arange(eigens.shape[0])

    for epoch in range(epochs):
        np.random.shuffle(indices)

        for i in range(0, indices.size, batch_size):
            eigens_batch = eigens[indices[i:i+batch_size]]
            labels_batch = labels[indices[i:i+batch_size]]

            yield epoch, step, eigens_batch, labels_batch

            step += 1


def random_shift(eigens, num_pixels):
    """
    """
    base_x = np.random.randint(4)
    base_y = np.random.randint(4)

    eigens = np.pad(
        eigens,
        ((0, 0), (2, 2), (2, 2), (0, 0)),
        mode='constant',
        constant_values=0)

    return eigens[:, base_x:base_x + 28, base_y:base_y + 28, :]


def test(model, issue_batches):
    """
    return accuracy on test set
    """
    session = tf.get_default_session()

    num_correct = 0
    num_predict = 0

    for epoch, step, eigens, labels in issue_batches:
        feeds = {
            model['eigens']: eigens,
        }

        guess = session.run(model['guess'], feed_dict=feeds)

        num_predict += guess.shape[0]
        num_correct += \
            np.sum(np.argmax(labels, axis=1) == np.argmax(guess, axis=1))

    return float(num_correct) / float(num_predict)


def train():
    """
    """
    datasets = load_datasets()

    train_batches = mnist_batches(
        datasets['train_eigens'], datasets['train_labels'], FLAGS.batch_size)

    model = build_capsnet()

    tested_epoch = 0

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch, step, eigens, labels in train_batches:
            feeds = {
                model['eigens']: random_shift(eigens, 2),
                model['labels']: labels,
                model['learning_rate']: 0.001 * (0.96 ** epoch),
            }

            fetch = {
                'loss': model['loss'],
                'trainer': model['trainer'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            if tested_epoch != epoch:
                tested_epoch = epoch

                issue_batches = mnist_batches(
                    datasets['issue_eigens'],
                    datasets['issue_labels'],
                    FLAGS.batch_size,
                    1)

                accuracy = test(model, issue_batches)

                print 'acc === [{:>4}][{:>5}]: {:>10,.8f}'.format(
                    epoch, step, accuracy)

            if step % 100 == 0:
                print 'loss[{:>4}][{:>5}]: {:>10,.8f}'.format(
                    epoch, step, fetched['loss'])


def main(_):
    """
    """
    train()


if __name__ == '__main__':
    tf.app.run()
