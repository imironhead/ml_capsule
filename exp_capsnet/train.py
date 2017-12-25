"""
"""
import numpy as np
import os
import tensorflow as tf

from six.moves import range
from mnist import load_mnist

from capsnet import build_capsnet

TFDEF = tf.app.flags
FLAGS = tf.app.flags.FLAGS


TFDEF.DEFINE_string('mnist-root-path', None, '')
TFDEF.DEFINE_string('affnist-test-path', None, '')
TFDEF.DEFINE_string('ckpt-path', None, '')
TFDEF.DEFINE_string('ckpt-dir', None, '')
TFDEF.DEFINE_string('logs-dir', None, '')

TFDEF.DEFINE_boolean('reconstruction-loss', False, '')
TFDEF.DEFINE_boolean('embed-positions', False, '')

TFDEF.DEFINE_integer('batch-size', 128, '')
TFDEF.DEFINE_integer('routing-frequency', 3, '')


def load_datasets():
    """
    load mnist
    """
    path_mnist = FLAGS.mnist_root_path
    path_affnist_issue = FLAGS.affnist_test_path

    path_train_eigens = os.path.join(path_mnist, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_mnist, 'train-labels-idx1-ubyte.gz')

    return load_mnist(
        path_train_eigens, path_train_labels, path_affnist_issue)


def mnist_batches(eigens, labels, batch_size, epochs=1000000):
    """
    batch data generator
    """
    step = 0

    indices = np.arange(eigens.shape[0])

    for epoch in range(epochs):
        np.random.shuffle(indices)

        for i in range(0, indices.size, batch_size):
            # NOTE: treate as training set
            if i + batch_size > indices.size and epochs > 1:
                break

            eigens_batch = eigens[indices[i:i+batch_size]]
            labels_batch = labels[indices[i:i+batch_size]]

            yield epoch, step, eigens_batch, labels_batch

            step += 1


def random_shift(eigens):
    """
    """
    if eigens.shape[1] != 28:
        raise 'only for mnist (28 x 28)'

    batch_size = eigens.shape[0]

    shifted_eigens = np.zeros((batch_size, 40, 40, 1))

    for i in range(batch_size):
        base_x = np.random.randint(12)
        base_y = np.random.randint(12)

        shifted_eigens[i, base_y:base_y+28, base_x:base_x+28] = eigens[i]

    return shifted_eigens


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


def main(_):
    """
    """
    reporter = tf.summary.FileWriter(FLAGS.logs_dir)

    if FLAGS.ckpt_dir is not None and tf.gfile.Exists(FLAGS.ckpt_dir):
        ckpt_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')

    datasets = load_datasets()

    train_batches = mnist_batches(
        datasets['train_eigens'], datasets['train_labels'], FLAGS.batch_size)

    model = build_capsnet()

    tested_epoch = 0

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if FLAGS.ckpt_path is not None and tf.gfile.Exists(FLAGS.ckpt_path):
            tf.train.Saver().restore(session, FLAGS.ckpt_path)

        for epoch, step, eigens, labels in train_batches:
            learning_rate = 0.001 * (0.95 ** epoch)

            feeds = {
                model['eigens']: random_shift(eigens),
                model['labels']: labels,
                model['learning_rate']: learning_rate,
            }

            fetch = {
                'loss': model['loss'],
                'step': model['training_step'],
                'trainer': model['trainer'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            # NOTE: general d_summaries
            summaries = [
                tf.Summary.Value(tag='loss', simple_value=fetched['loss']),
                tf.Summary.Value(tag='lr', simple_value=learning_rate)]

            summaries = tf.Summary(value=summaries)

            reporter.add_summary(summaries, fetched['step'])

            if step % 100 == 0:
                print 'loss[{:>8}]: {}'.format(step, fetched['loss'])

            # NOTE: new epoch
            if tested_epoch != epoch:
                tested_epoch = epoch

                issue_batches = mnist_batches(
                    datasets['issue_eigens'],
                    datasets['issue_labels'],
                    FLAGS.batch_size,
                    1)

                accuracy = test(model, issue_batches)

                print 'accuracy[{:>8}]: {}'.format(epoch, accuracy)

                # TODO: summary
                summary_accuracy = \
                    [tf.Summary.Value(tag='accuracy', simple_value=accuracy)]

                summary = tf.Summary(value=summary_accuracy)

                reporter.add_summary(summary, fetched['step'])

                # NOTE: save model
                if epoch % 10 == 0 and ckpt_path is not None:
                    tf.train.Saver().save(
                        session, ckpt_path, global_step=model['training_step'])

                if epoch > 200:
                    break


if __name__ == '__main__':
    tf.app.run()
