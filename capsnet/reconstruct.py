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

    return load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)


def save(
        path, right_inputs, right_outputs, wrong_inputs, wrong_outputs):
    """
    right_inputs: images that are predicted correctly
    right_outputs: reconstructions of right_inputs
    wrong_inputs: images that are predicted incorrectly
    wrong_outputs: reconstructions of wrong_inputs
    """
    temp_images = np.zeros((64, 28, 28, 1))

    right_inputs = np.concatenate(
        [right_inputs, temp_images[:64-right_inputs.shape[0]]], axis=0)
    right_outputs = np.concatenate(
        [right_outputs, temp_images[:64-right_outputs.shape[0]]], axis=0)
    wrong_inputs = np.concatenate(
        [wrong_inputs, temp_images[:64-wrong_inputs.shape[0]]], axis=0)
    wrong_outputs = np.concatenate(
        [wrong_outputs, temp_images[:64-wrong_outputs.shape[0]]], axis=0)

    # NOTE: to 2 (128 * 28, 28) images
    i_images = np.concatenate([right_inputs, wrong_inputs], axis=0)
    o_images = np.concatenate([right_outputs, wrong_outputs], axis=0)

    i_images = i_images.reshape((-1, 28))
    o_images = o_images.reshape((-1, 28))

    # NOTE: to 128 image pairs: ((28, 28), (28, 28))
    images = zip(
        np.split(i_images, 128, axis=0), np.split(o_images, 128, axis=0))

    # NOTE: to 128 images: (56, 28)
    images = [np.concatenate(pair, axis=0) for pair in images]

    # NOTE: to 4 images: (56, 28 * 32)
    images = \
        [np.concatenate(images[i:i+32], axis=1) for i in range(0, 128, 32)]

    # NOTE: stitch a white splitter
    temp, images = images, []

    for image in temp:
        image = np.concatenate([image, np.ones((1, 32 * 28))], axis=0)

        images.append(image)

    # NOTE: to the final image
    images = np.concatenate(images, axis=0)

    images = images * 255.0

    images = np.clip(images, 0.0, 255.0).astype(np.uint8)

    scipy.misc.imsave(path, images)


def main(_):
    """
    """
    FLAGS.mnist_root_path = '/home/ironhead/datasets/mnist'
    FLAGS.ckpt_path = './ckpt/9600/model.ckpt-18761'
    FLAGS.meta_path = './ckpt/9600/model.ckpt-18761.meta'
    FLAGS.result_path = './qooo.png'
    FLAGS.sample_number = 256

    datasets = load_datasets()

    FLAGS.sample_number = ((FLAGS.sample_number + 15) / 16) * 16

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(FLAGS.meta_path)

        saver.restore(session, FLAGS.ckpt_path)

        graph = tf.get_default_graph()

        images_tensor = graph.get_tensor_by_name('images:0')
        labels_tensor = graph.get_tensor_by_name('labels:0')
        prediction_tensor = graph.get_tensor_by_name('predictions:0')
        reconstruction_tensor = graph.get_tensor_by_name('fc_784/Sigmoid:0')

        reconstructions = np.zeros_like(datasets['issue_eigens'])
        predictions = np.zeros_like(datasets['issue_labels'])

        for i in range(0, datasets['issue_eigens'].shape[0], 128):
            images = datasets['issue_eigens'][i:i+128]
            labels = datasets['issue_labels'][i:i+128]

            feeds = {
                images_tensor: images,
                labels_tensor: labels,
            }

            fetch = {
                'predictions': prediction_tensor,
                'reconstructions': reconstruction_tensor,
            }

            fetched = session.run(fetch, feed_dict=feeds)

            predictions[i:i+128] = fetched['predictions']
            reconstructions[i:i+128] = \
                fetched['reconstructions'].reshape((-1, 28, 28, 1))

    truth = np.argmax(datasets['issue_labels'], axis=1)
    guess = np.argmax(predictions, axis=1)

    right = np.argwhere(truth == guess).flatten()
    wrong = np.argwhere(truth != guess).flatten()

    save(
        FLAGS.result_path,
        datasets['issue_eigens'][right[:64]],
        reconstructions[right[:64]],
        datasets['issue_eigens'][wrong[:64]],
        reconstructions[wrong[:64]])


if __name__ == '__main__':
    tf.app.run()
