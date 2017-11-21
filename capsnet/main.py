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

tf.app.flags.DEFINE_integer(
    'batch-size', 128, '')


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


def mnist_batches(eigens, labels, batch_size):
    """
    batch data generator
    """
    epoch, step = -1, -1

    indices = np.arange(eigens.shape[0])

    while True:
        epoch, step = epoch + 1, 0

        np.random.shuffle(indices)

        for i in range(0, indices.size, batch_size):
            if i + batch_size > indices.size:
                break

            eigens_batch = eigens[indices[i:i+batch_size]]
            labels_batch = labels[indices[i:i+batch_size]]

            yield epoch, step, eigens_batch, labels_batch

            step += 1


def test(model, issue_batches):
    """
    return accuracy on test set
    """
    session = tf.get_default_session()

    num_correct = 0
    num_predict = 0

    for epoch, step, eigens, labels in issue_batches:
        if epoch > 0:
            break

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

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch, step, eigens, labels in train_batches:
            feeds = {
                model['eigens']: eigens,
                model['labels']: labels,
            }

            fetch = {
                'loss': model['loss'],
                'trainer': model['trainer'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            if step == 0:
                issue_batches = mnist_batches(
                    datasets['issue_eigens'],
                    datasets['issue_labels'],
                    FLAGS.batch_size)

                accuracy = test(model, issue_batches)

                print 'acc [{:>4}][{:>5}]: {:>10,.8f}'.format(
                    epoch, step, accuracy)

            if step % 100 == 0:
                print 'loss[{:>4}][{:>5}]: {:>10,.8f}'.format(
                    epoch, step, fetched['loss'])


def main(_):
    """
    """
    FLAGS.batch_size = 128
    FLAGS.mnist_root_path = '/home/ironhead/datasets/mnist'

    train()


if __name__ == '__main__':
    tf.app.run()
