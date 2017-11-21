"""
"""
import tensorflow as tf


def margin_loss(labels, logits):
    """
    arXiv:1710.09829v1, #3: margin loss for digit existence

    labels.shape -> (-1, 10)
    logits.shape -> (-1, 10)
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    a = tf.maximum(0.0, m_plus - logits) ** 2
    b = tf.maximum(0.0, logits - m_minus) ** 2

    loss = labels * a + lambda_ * (1.0 - labels) * b

    # the total loss is simply the sum of the losses of all digit capsules.
    loss = tf.reduce_sum(loss, axis=1)

    return tf.reduce_mean(loss)


def squash(tensor):
    """
    arXiv:1710.09829v1, #2: squashing
    """
    lensqr = tf.reduce_sum(tensor ** 2, axis=-1, keep_dims=True)
    length = tf.sqrt(lensqr + 1e-9)

    result = tensor * (lensqr / (lensqr + 1.0) / length)

    return result


def routing(uhat):
    """
    """


def build_capsnet():
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
