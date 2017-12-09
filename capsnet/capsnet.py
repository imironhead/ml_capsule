"""
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


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


def reconstruction_loss(labels, digit_capsules, images):
    """
    """
    digit_capsules = tf.reshape(digit_capsules, (-1, 10, 16))

    images = tf.reshape(images, (-1, 784))

    labels = tf.reshape(labels, (-1, 10, 1))

    flow = tf.reshape(digit_capsules * labels, (-1, 160))

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    for index, num_outputs in enumerate([512, 1024, 784]):
        flow = tf.contrib.layers.fully_connected(
            inputs=flow,
            num_outputs=num_outputs,
            activation_fn=tf.nn.sigmoid if index == 2 else tf.nn.relu,
            weights_initializer=initializer,
            scope='fc_{}'.format(num_outputs))

    sqr_diff = (images - flow) ** 2

    return 0.0005 * tf.reduce_sum(sqr_diff, axis=1)


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

    # NOTE: arXiv:1710.09829v1, #5: Capsules on MNIST
    #       training is performed on 28x28 MNIST images that have been shifted
    #       by up to 2 pixels in each direction with zero padding.
    images = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)

    batch_size = tf.shape(images)[0]

    # NOTE: arXiv:1710.09829v1, #4: CapsNet architecture
    #       this layer converts pixel intensities to the activities of local
    #       feature detectors that are then used as inputs to the primary
    #       capsules.
    # NOTE: shape -> (-1, 20, 20, 256)
    conv1 = tf.layers.conv2d(
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

    # NOTE: arXiv:1710.09829v1, #4: CapsNet architecture
    #       the second layer (PrimaryCapsules) is a convolutional capsule layer
    #       with 32 channels of convolutional 8D capsules (i.e. each primary
    #       capsule contains 8 convolutional units with a 9x9 kernel and a
    #       stride of 2).
    # NOTE: shape -> (-1, 6, 6, 8 * 32)
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

    # NOTE: shape -> (-1, 36 * 32, 1, 8)
    #       each primary capsule is a (1, 8) matrix
    primary_capsules = tf.reshape(primary_capsules, [-1, 36 * 32, 1, 8])

    # NOTE: arXiv:1710.09829v1, #4: CapsNet architecture
    #       we want the length of the output vector of the capsule to represent
    #       the probability that the entity represented by the capsule is
    #       present in the current input.
    # NOTE: arXiv:1710.09829v1, #4: CapsNet architecture
    #       we therefore use a non-linear "squashing" function to ensure that
    #       short vectors get shrunk to almost zero length and long vectors get
    #       shrunk to a length slightly below 1.
    primary_capsules = squash(primary_capsules)

    # NOTE: 36 * 32 primary capsules to 10 digit capsules.
    #       each digit capsule is weighted sum of all primary capsules.
    w = tf.get_variable(
        'w',
        [1, 36 * 32, 8, 16 * 10],
        trainable=True,
        initializer=initializer,
        dtype=tf.float32)

    # NOTE: share weights accross batch
    weights = tf.tile(w, [tf.shape(primary_capsules)[0], 1, 1, 1])

    # NOTE: shape -> (-1, 36 * 32, 1, 16 * 10)
    #       the latest dimension (16*10) is composed with ten 16D vectors
    uhat = tf.matmul(primary_capsules, weights)

    uhat = tf.reshape(uhat, [-1, 36 * 32, 10, 16])

    # NOTE: shape -> [-1, 10, 36 * 32, 16]
    #       each digit capsule (10) is composed with 36 * 32 16D weighted
    #       vectors
    uhat = tf.transpose(uhat, [0, 2, 1, 3])

    # NOTE: each primary capsule (36 * 32) contributes to 10 digit capsules.
    b = tf.zeros(shape=(batch_size, 10, 1, 36 * 32))

    # NOTE: gradients from uhat_stop stop here (no backpropagation)
    uhat_stop = tf.stop_gradient(uhat, name='uhat_stop')

    # NOTE: arXiv:1710.09829v1, #2: how the vector inputs and outputs of a
    #       capsule are computed
    for r in reversed(range(FLAGS.routing_frequency + 1)):
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
            # NOTE: arXiv:1710.09829v1, #2: how the vector inputs and outputs
            #       of a capsule are computed
            #       the agreement is simply the scalar product
            #       a_ij = v_j dot u_hat_ji
            #       this agreement is treated as if it were a log likelihood
            #       and is added to the initial logit, b_ij before computing
            #       the new values for all the coupling coefficients linking
            #       capsule i to higher level capsules
            b += tf.matmul(v, uhat_stop, transpose_b=True)

    digit_capsules = v

    # NOTE: arXiv:1710.09829v1, #3: margin loss for digit existence
    #       we are using the length of the instantiation vector to represent
    #       the probability that a capsule's entity exists, so we would like
    #       the top-level capsule for digit class k to have a long
    #       instantiation vector if and only if that digit is present in the
    #       image.
    guess = tf.norm(digit_capsules, ord=2, axis=3)

    guess = tf.reshape(guess, [-1, 10])

    loss = margin_loss(labels, guess)

    # NOTE:
    if FLAGS.reconstruction_loss:
        loss += reconstruction_loss(labels, digit_capsules, images)

    loss = tf.reduce_mean(loss)

    # NOTE: arXiv:1710.09829v1, #4capsnet architecture
    #       we use the adam optimizer with its tensorflow default parameters
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss)

    return {
        'eigens': images,
        'labels': labels,
        'loss': loss,
        'guess': guess,
        'trainer': trainer,
        'learning_rate': learning_rate,
    }
