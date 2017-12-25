"""
"""
import gzip
import numpy as np
import scipy.io


def read32(bytestream):
    """
    read a 32 bit unsigned int fron the bytestream.
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(path_file):
    """
    extract the images into a numpy array in shape [number, y, x, depth].
    """
    print('extracting', path_file)

    with gzip.open(path_file) as bytestream:
        magic = read32(bytestream)

        if magic != 2051:
            raise Exception('invalid mnist data: {}'.format(path_file))

        size = read32(bytestream)
        rows = read32(bytestream)
        cols = read32(bytestream)

        buff = bytestream.read(size * rows * cols)
        data = np.frombuffer(buff, dtype=np.uint8)
        data = data.reshape(size, rows, cols, 1)

    return data


def extract_labels(path_file):
    """
    extract the labels into a numpy array with shape [index].
    """
    print('extracting', path_file)

    with gzip.open(path_file) as bytestream:
        magic = read32(bytestream)

        if magic != 2049:
            raise Exception('invalid mnist data: {}'.format(path_file))

        size = read32(bytestream)
        buff = bytestream.read(size)
        labels = np.frombuffer(buff, dtype=np.uint8)

    return labels.astype(np.int32)


def load_mnist(
        path_mnist_train_eigens,
        path_mnist_train_labels,
        path_affnist_issue):
    """
    """
    train_eigens = extract_images(path_mnist_train_eigens)
    train_labels = extract_labels(path_mnist_train_labels)

    # to 0.0 ~ +1.0
    train_eigens = train_eigens.astype(np.float32) / 255.0

    # one hot labels
    train_labels_onehot = np.zeros((train_labels.size, 10))

    train_labels_onehot[np.arange(train_labels.size), train_labels] = 1.0

    # affNIST
    aff_nist_issue = scipy.io.matlab.loadmat(path_affnist_issue)

    # NOTE: already one-hot labels in affnist
    issue_labels_onehot = np.transpose(aff_nist_issue['affNISTdata'][0, 0][4])

    # NOTE: affnist = {
    #           'affNISTdata': [[
    #               transforms_type_0,
    #               transforms_type_1,
    #               images,
    #               indices_0,
    #               onehot_labels,
    #               labels,
    #               transforms_type_2,
    #               indices_1,
    #           ]]
    #       }
    issue_eigens = np.transpose(aff_nist_issue['affNISTdata'][0, 0][2])

    issue_eigens = np.reshape(issue_eigens, (-1, 40, 40, 1))

    # to 0.0 ~ +1.0
    issue_eigens = issue_eigens.astype(np.float32) / 255.0

    return {
        'train_eigens': train_eigens,
        'train_labels': train_labels_onehot,
        'issue_eigens': issue_eigens,
        'issue_labels': issue_labels_onehot,
    }


if __name__ == '__main__':
    import os

    path_root = '/home/ironhead/datasets/mnist'

    path_train_eigens = os.path.join(path_root, 'train-images-idx3-ubyte.gz')
    path_train_labels = os.path.join(path_root, 'train-labels-idx1-ubyte.gz')
    path_issue_eigens = os.path.join(path_root, 't10k-images-idx3-ubyte.gz')
    path_issue_labels = os.path.join(path_root, 't10k-labels-idx1-ubyte.gz')

    dataset = load_mnist(
        path_train_eigens, path_train_labels,
        path_issue_eigens, path_issue_labels)

    print dataset['train_eigens'].shape
    print dataset['issue_eigens'].shape

    print np.sum(dataset['train_eigens'][:, :2])
    print np.sum(dataset['train_eigens'][:, -2:])

    print np.sum(dataset['train_eigens'][:, :, :2])
    print np.sum(dataset['train_eigens'][:, :, -2:])

    print np.sum(dataset['issue_eigens'][:, :2])
    print np.sum(dataset['issue_eigens'][:, -2:])

    print np.sum(dataset['issue_eigens'][:, :, :2])
    print np.sum(dataset['issue_eigens'][:, :, -2:])
