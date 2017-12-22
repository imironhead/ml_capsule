"""
"""
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def capsulize(
        tensor,
        kernel_size,
        strides,
        out_capsule_layer_num,
        out_capsule_dim,
        embed_positions,
        scope):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        capsules = tf.layers.conv2d(
            tensor,
            filters=out_capsule_dim * out_capsule_layer_num,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            use_bias=False,
            kernel_initializer=initializer,
            name='capsulize')

        batch_size, h, w, c = capsules.shape
        batch_size = tf.shape(capsules)[0]

        # NOTE: each capsule is a (1, out_capsule_dim) matrix
        capsules = tf.reshape(
            capsules,
            [-1, h * w * out_capsule_layer_num, 1, out_capsule_dim])

        if embed_positions:
            # TODO: maintain stddev / mean
            positions = np.ones((w, h, 2 * out_capsule_layer_num))

            positions[:, :, 1::2] = np.arange(h)

            positions = np.transpose(positions, [1, 0, 2])

            positions[:, :, 0::2] = np.arange(w)

            # TODO: stddev -> 0.02, mean -> 0.0

            positions = np.reshape(
                positions,
                [1, h * w * out_capsule_layer_num, 1, out_capsule_dim])

            # to tensor
            positions = tf.constant(positions)

            # tile
            positions = tf.tile(positions, [batch_size, 1, 1, 1])

            # concat
            capsules = tf.concat([capsules, positions], axis=3)

        capsules = squash(capsules)

    return capsules


def route(
        capsules, routing_frequency, out_capsule_num, out_capsule_dim, scope):
    """
    capsules: [-1, in_capsule_num, 1, in_capsule_dim]
    """
    batch_size, in_capsule_num, _, in_capsule_dim = capsules.shape
    batch_size = tf.shape(capsules)[0]

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # NOTE: 36 * 32 primary capsules to 10 digit capsules.
        #       each digit capsule is weighted sum of all primary capsules.
        w = tf.get_variable(
            'w',
            [1, in_capsule_num, in_capsule_dim,
                out_capsule_num * out_capsule_dim],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        # NOTE: share weights accross batch
        weights = tf.tile(w, [batch_size, 1, 1, 1])

        # NOTE: shape -> (-1, 36 * 32, 1, 16 * 10)
        #       the latest dimension (16*10) is composed with ten 16D vectors
        uhat = tf.matmul(capsules, weights)

        uhat = tf.reshape(
            uhat, [-1, in_capsule_num, out_capsule_num, out_capsule_dim])

        # NOTE: shape -> [-1, 10, 36 * 32, 16]
        #       each digit capsule (10) is composed with 36 * 32 16D weighted
        #       vectors
        uhat = tf.transpose(uhat, [0, 2, 1, 3])

        # NOTE: each primary capsule (36 * 32) contributes to 10 digit
        #       capsules.
        b = tf.zeros(shape=(batch_size, out_capsule_num, 1, in_capsule_num))

        # NOTE: gradients from uhat_stop stop here (no backpropagation)
        uhat_stop = tf.stop_gradient(uhat, name='uhat_stop')

        # NOTE: arXiv:1710.09829v1, #2: how the vector inputs and outputs of a
        #       capsule are computed
        for r in reversed(range(routing_frequency + 1)):
            # NOTE: routing softmax
            #       probabilities of each primary capsule's contribution to 10
            #       digit capsules
            c = tf.nn.softmax(b, dim=1)

            # NOTE: compose digit capsules
            #       use uhat_stop if we have to make agreement
            s = tf.matmul(c, uhat if r == 0 else uhat_stop)

            # NOTE: shape -> [-1, 10, 1, 16]
            v = squash(s)

            if r > 0:
                # NOTE: arXiv:1710.09829v1, #2: how the vector inputs and
                #       outputs of a capsule are computed
                #       the agreement is simply the scalar product
                #       a_ij = v_j dot u_hat_ji
                #       this agreement is treated as if it were a log
                #       likelihood and is added to the initial logit,
                #       b_ij before computing the new values for all the
                #       coupling coefficients linking capsule i to higher
                #       level capsules
                b += tf.matmul(v, uhat_stop, transpose_b=True)

        out_capsules = tf.reshape(
            v, [-1, out_capsule_num, out_capsule_dim], name='output_capsules')

        return out_capsules


def squash(tensor):
    """
    arXiv:1710.09829v1, #2: squashing
    """
    lensqr = tf.reduce_sum(tensor ** 2, axis=-1, keep_dims=True)
    length = tf.sqrt(lensqr + 1e-9)

    return tensor * (lensqr / (lensqr + 1.0) / length)


def fully_connected_reconstruction(capsules, labels, layers):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    batch_size, capsule_num, capsule_dim = capsules.shape
    batch_size = tf.shape(capsules)[0]

    with tf.variable_scope('reconstruction', reuse=tf.AUTO_REUSE):
        labels = tf.reshape(labels, (-1, capsule_num, 1))

        tensor = tf.reshape(capsules * labels, (-1, capsule_num * capsule_dim))

        for index, num_outputs in enumerate(layers):
            if index + 1 == len(layers):
                activation_fn = tf.nn.sigmoid
            else:
                activation_fn = tf.nn.relu

            tensor = tf.contrib.layers.fully_connected(
                inputs=tensor,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                weights_initializer=initializer,
                scope='fc_{}'.format(index))

    return tensor


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

    # NOTE: arXiv:1710.09829v1, #3: margin loss for digit existence
    #       the total loss is simply the sum of the losses of all digit
    #       capsules.
    return tf.reduce_sum(loss, axis=1)


def build_capsnet():
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    labels = tf.placeholder(
        shape=[None, 10], dtype=tf.float32, name='labels')

    images = tf.placeholder(
        shape=[None, 28, 28, 1], dtype=tf.float32, name='images')

    tensor = tf.layers.conv2d(
        images,
        filters=256,
        kernel_size=9,
        strides=1,
        padding='valid',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv1')

    capsules = capsulize(tensor, 9, 2, 32, 8, False, 'capsules_1')

    capsules = route(capsules, 3, 10, 16, 'capsules_2')

    guess = tf.norm(capsules, ord=2, axis=2)

    loss = margin_loss(labels, guess)

    # NOTE
    new_images = fully_connected_reconstruction(
        capsules, labels, [512, 1024, 784])

    old_images = tf.reshape(images, (-1, 28 * 28))

    sqr_diff = tf.square(old_images - new_images)

    loss += 0.0005 * tf.reduce_sum(sqr_diff, axis=1)

    loss = tf.reduce_mean(loss)

    # NOTE: arXiv:1710.09829v1, #4capsnet architecture
    #       we use the adam optimizer with its tensorflow default parameters
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    training_step = tf.get_variable(
        'training_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=training_step)

    return {
        'eigens': images,
        'labels': labels,
        'loss': loss,
        'guess': guess,
        'trainer': trainer,
        'learning_rate': learning_rate,
        'training_step': training_step,
    }
