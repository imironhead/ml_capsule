"""
"""
import tensorflow as tf


def build_basenet():
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
