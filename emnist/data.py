
import gzip
import numpy as np
import random


def load_training_images(filename='./gzip/emnist-digits-train-images-idx3-ubyte.gz'):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        f.close()
        return images


def load_training_labels(filename='./gzip/emnist-digits-train-labels-idx1-ubyte.gz'):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        f.close()
        return labels


def load_test_images(filename='./gzip/emnist-digits-test-images-idx3-ubyte.gz'):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        f.close()
        return images


def load_test_labels(filename='./gzip/emnist-digits-test-labels-idx1-ubyte.gz'):
    with gzip.open(filename, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        f.close()
        return labels


def load_training_data(batch_size=10000, shuffle=False):
    _data = load_training_images()
    print(f"Found {len(_data)} Images.")
    _labels = load_training_labels()
    start = 0
    idx = 0
    if(shuffle):
        # Source: https://stackoverflow.com/a/23289591
        tmp = list(zip(_data, _labels))
        random.shuffle(tmp)
        _data, _labels = zip(*tmp)

    while start < len(_data):
        end = start + batch_size
        print(f"Batch Processing: start: {start}, end: {end}")
        labels = []
        targets = []
        inputs = []
        if(batch_size <= len(_data)):
            for data in _data[start:end]:
                if(idx < start):
                    raise Exception('you should take a look..')
                labels.append(_labels[idx])
                targets.append([float(0) if index != _labels[idx] else float(1)
                                for index in range(10)])
                inputs.append([round(float(c/255), 2)
                              for _ in data for c in _])
                idx += 1
                b = "Processing file #" + str(idx)
                print(b, end="\r")
        print()
        yield labels, targets, inputs
        start += batch_size


def load_test_data():
    _data = load_test_images()
    print(f"Found {len(_data)} Images.")
    _labels = load_test_labels()

    idx = 0
    labels = []
    targets = []
    inputs = []
    for data in _data:
        labels.append(_labels[idx])
        targets.append([float(0) if index != _labels[idx] else float(1)
                        for index in range(10)])
        inputs.append([round(float(c/255), 2)
                       for _ in data for c in _])
        idx += 1
        b = "Processing file #" + str(idx)
        print(b, end="\r")
    print()
    return labels, targets, inputs
