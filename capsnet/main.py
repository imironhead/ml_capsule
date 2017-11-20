"""
"""
import numpy as np
import os
import tensorflow as tf

from six.moves import range
from mnist import load_mnist


def mnist_batch(eigens, labels, batch_size):
    """
    """
    epoch, step = 0, 0

    indices = np.arange(eigens.shape[0])

    while True:
        for i in range(0, indices.size, batch_size):
            if i + batch_size > indices.size:
                break

            eigens_batch = eigens[indices[i:i+batch_size]]
            labels_batch = labels[indices[i:i+batch_size]]

            yield epoch, step, eigens_batch, labels_batch

            step += 1

        np.random.shuffle(indices)

        epoch += 1


def margin_loss(labels, logits):
    """
    """
    a = tf.maximum(0.0, 0.9 - logits) ** 2
    b = tf.maximum(0.0, logits - 0.1) ** 2

    loss = labels * a + 0.5 * (1.0 - labels) * b

    loss = tf.reduce_sum(loss, axis=1)

    return tf.reduce_mean(loss)


def squash(tensor):
    """
    """
    lensqr = tf.reduce_sum(tensor ** 2, axis=-1, keep_dims=True)
    length = tf.sqrt(lensqr + 1e-9)

    result = tensor * (lensqr / (lensqr + 1.0) / length)

    return result


def routing(uhat):
    """
    """


def build_baseline_network():
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    images = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)

    labels = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    # batch_size = images.shape[0]

    cropped_images = \
        tf.random_crop(images, size=[128, 28, 28, 1])

    # arXiv:1710.09829v1
    # this layer converts pixel intensities to the activities of local feature
    # detectors that are then used as inputs to the primary capsules.
    conv1 = tf.layers.conv2d(
        cropped_images,
        filters=256,
        kernel_size=9,
        strides=1,
        padding='valid',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv1')

    #
    conv2 = tf.layers.conv2d(
        conv1,
        filters=8 * 32,
        kernel_size=9,
        strides=2,
        padding='valid',
        data_format='channels_last',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv2')

    flow = tf.contrib.layers.flatten(conv2)

    flow = tf.contrib.layers.fully_connected(
        inputs=flow,
        num_outputs=10,
        activation_fn=None,
        weights_initializer=initializer,
        scope='fc_end')

    guess = tf.nn.sigmoid(flow)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=flow)

    loss = tf.reduce_mean(loss)

    trainer = tf.train \
        .AdamOptimizer() \
        .minimize(loss)

    return {
        'eigens': images,
        'labels': labels,
        'loss': loss,
        'guess': guess,
        'trainer': trainer,
    }


def build_network():
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    labels = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    images = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)

    cropped_images = \
        tf.random_crop(images, size=[128, 28, 28, 1])

    # arXiv:1710.09829v1
    # this layer converts pixel intensities to the activities of local feature
    # detectors that are then used as inputs to the primary capsules.
    conv1 = tf.layers.conv2d(
        cropped_images,
        filters=256,
        kernel_size=9,
        strides=1,
        padding='valid',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv1')

    #
    primary_capsules = tf.layers.conv2d(
        conv1,
        filters=8 * 32,
        kernel_size=9,
        strides=2,
        padding='valid',
        data_format='channels_last',
        # activation=None,
        # activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        name='primary_capsules')

    primary_capsules = tf.reshape(primary_capsules, [-1, 36 * 32, 1, 8])

    primary_capsules = squash(primary_capsules)

    print primary_capsules.shape

    w = tf.get_variable(
        'w',
        [1, 36 * 32, 8, 16 * 10],
        trainable=True,
        initializer=initializer,
        dtype=tf.float32)

    w = tf.tile(w, [tf.shape(primary_capsules)[0], 1, 1, 1])

    u = primary_capsules

    uhat = tf.matmul(u, w)

    # [-1, 36 * 32, 1, 16 * 10]
    print uhat.shape

    uhat = tf.reshape(uhat, [-1, 36 * 32, 10, 16])

    # [-1, 10, 36 * 32, 16]
    uhat = tf.transpose(uhat, [0, 2, 1, 3])

    # b = tf.zeros(shape=(1, 10, 36 * 32))
    #
    # b = tf.tile(b, [tf.shape(primary_capsules)[0], 1, 1])

    b = tf.zeros(shape=(128, 10, 1, 36 * 32))

    uhat_stop = tf.stop_gradient(uhat, name='ooxx')

    for r in reversed(range(4)):
        with tf.variable_scope('iii_{}'.format(r)):
            xhat = uhat if r == 0 else uhat_stop

            c = tf.nn.softmax(b, dim=1)

            s = tf.matmul(c, xhat)

            # [-1, 1152, 1, 16]
            v = squash(s)

            if r > 0:
                db = tf.matmul(v, xhat, transpose_b=True)

                b = b + db

                print 'dbdb: {}'.format(db.shape)

    # [-1, 10, 1, 16]
    print v.shape

    v_norm = tf.norm(v, ord=2, axis=3)

    v_norm = tf.reshape(v_norm, [-1, 10])

    loss = margin_loss(labels, v_norm)

    trainer = tf.train \
        .AdamOptimizer() \
        .minimize(loss)

    return {
        'eigens': images,
        'labels': labels,
        'loss': loss,
        'guess': v_norm,
        'trainer': trainer,
    }


def train(model, dataset):
    """
    """
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        batches = mnist_batch(
            dataset['train_eigens'], dataset['train_labels'], 128)

        for epoch, step, eigens, labels in batches:
            feeds = {
                model['eigens']: eigens,
                model['labels']: labels,
            }

            fetch = {
                'loss': model['loss'],
                'trainer': model['trainer'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            if step % 100 == 0:
                print 'loss[{}]: {}'.format(step, fetched['loss'])

            if step % 1000 == 0:
                test(model, dataset)
                # break


def test(model, dataset):
    """
    """
    session = tf.get_default_session()

    batches = mnist_batch(
        dataset['issue_eigens'], dataset['issue_labels'], 128)

    num_correct = 0
    num_predict = 0

    for epoch, step, eigens, labels in batches:
        if epoch > 0:
            break

        feeds = {
            model['eigens']: eigens,
        }

        guess = session.run(model['guess'], feed_dict=feeds)

        num_predict += guess.shape[0]
        num_correct += \
            np.sum(np.argmax(labels, axis=1) == np.argmax(guess, axis=1))

    print "accuracy: {}".format(float(num_correct) / float(num_predict))


if __name__ == '__main__':
    """
    """
    path_root = '/home/ironhead/datasets/mnist'
    path_train_eigens = os.path.join(path_root, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_root, 'train-labels-idx1-ubyte.gz')
    path_issue_eigens = os.path.join(path_root, 't10k-images-idx3-ubyte.gz')
    path_issue_labels = os.path.join(path_root, 't10k-labels-idx1-ubyte.gz')

    dataset = load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)

    model = build_network()

    train(model, dataset)
