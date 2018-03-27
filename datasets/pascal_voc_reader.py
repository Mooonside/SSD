from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorflow.python.ops import image_ops
from functools import partial as set_parameter

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_DIR = '/home/yifeng/DataSets/VOCdevkit/VOC2012/tf_detects/tf_records/train'
VALIDATION_DIR = '/home/yifeng/DataSets/VOCdevkit/VOC2012/tf_detects/tf_records/val'
TRAIN_NUM = 5717
VALID_NUM = 5823


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/name': tf.FixedLenFeature([], tf.string),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(tf.int64)

        })

    features['image/encoded'] = image_ops.decode_jpeg(features['image/encoded'], channels=3)
    return features


def extract(features):
    name = features['image/name']
    image = features['image/encoded']
    xmins = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymins = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmaxs = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymaxs = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
    # read data in the format xmins, ymins, xmaxs, ymaxes !
    bboxes = tf.transpose(tf.stack([ymins, xmins, ymaxs, xmaxs]))

    return name, image, labels, bboxes


def augment(name, image, labels, bboxes):
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return name, image, labels, bboxes


def normalize(name, image, labels, bboxes):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return name, image, labels, bboxes


def cast_type(name, image, labels, bboxes):
    return name, tf.cast(image, tf.float32), tf.cast(labels, tf.int32), bboxes


def reshape(name, image, label, bboxes, reshape_size=None):
    if reshape_size is not None:
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bilinear(image, reshape_size)

    return name, tf.squeeze(image, axis=0), label, bboxes


def get_dataset(dir, batch_size, num_epochs, reshape_size=[300, 300], padding='SAME'):
    if not num_epochs:
        num_epochs = None
    filenames = [os.path.join(dir, i) for i in os.listdir(dir)]

    with tf.name_scope('input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        # could also be [list, of, filenames]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)
        dataset = dataset.map(extract)
        dataset = dataset.map(cast_type)
        dataset = dataset.map(augment)
        dataset = dataset.map(normalize)
        dataset = dataset.map(set_parameter(reshape, reshape_size=reshape_size))

        # the parameter is the queue size
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.batch(batch_size)
    return dataset


def get_next_batch(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == '__main__':
    dataset = get_dataset(dir=TRAIN_DIR, batch_size=1, num_epochs=1)
    name, image, labels, bboxes = get_next_batch(dataset)

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    name_v, image_v, labels_v, bboxes_v = sess.run([name, image, labels, bboxes])
    print(name_v)
    print(image_v.shape)
    print(labels_v.shape)
    print(bboxes_v.shape)
