import os
import random
import xml.etree.ElementTree as ET
import numpy as np


from datasets.pascal_voc_utils import pascal_voc_dict
import tensorflow as tf


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


DEFUALT_PATHS = {
    'images': '/home/yifeng/DataSets/VOCdevkit/VOC2012/JPEGImages',
    'annotations': '/home/yifeng/DataSets/VOCdevkit/VOC2012/Annotations'
}


class PascalVocWriter:
    """
        PASCAL VOC 2012 DataSet to TF record Writer
    """

    def __init__(self, paths=DEFUALT_PATHS):
        self.img_path = paths['images']
        self.ano_path = paths['annotations']

    def convert_to_example(self, file_name):
        img_path = os.path.join(self.img_path, file_name + '.jpg')
        ano_path = os.path.join(self.ano_path, file_name + '.xml')

        img_data = tf.gfile.FastGFile(img_path, 'rb').read()

        anno_tree = ET.parse(ano_path)
        anno_root = anno_tree.getroot()

        size = anno_root.find('size')
        shape = [int(size.find('height').text),
                 int(size.find('width').text),
                 int(size.find('depth').text)]

        image_format = b'JPEG'

        bboxes = []
        labels = []

        difficult = []
        truncated = []
        for obj in anno_root.findall('object'):
            label = obj.find('name').text
            labels.append(int(pascal_voc_dict[label]))

            if obj.find('difficult'):
                difficult.append(int(obj.find('difficult').text))
            else:
                difficult.append(0)

            if obj.find('truncated'):
                truncated.append(int(obj.find('truncated').text))
            else:
                truncated.append(0)

            bbox = obj.find('bndbox')
            bboxes.append((int(bbox.find('xmin').text),
                           int(bbox.find('ymin').text),
                           int(bbox.find('xmax').text),
                           int(bbox.find('ymax').text)
                           ))

        # turn to relative mode
        bboxes = np.asarray(bboxes, dtype=np.float32)
        xmins = (bboxes[:, 0] / shape[1]).tolist()
        ymins = (bboxes[:, 1] / shape[0]).tolist()
        xmaxs = (bboxes[:, 2] / shape[1]).tolist()
        ymaxs = (bboxes[:, 3] / shape[0]).tolist()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/name':bytes_feature(file_name.encode()),
                    'image/shape': int64_feature(shape),
                    'image/format': bytes_feature(image_format),
                    'image/encoded': bytes_feature(img_data),
                    'image/object/bbox/xmin': float_feature(xmins),
                    'image/object/bbox/ymin': float_feature(ymins),
                    'image/object/bbox/xmax': float_feature(xmaxs),
                    'image/object/bbox/ymax': float_feature(ymaxs),
                    'image/object/bbox/label': int64_feature(labels),
                    'image/object/bbox/difficult': int64_feature(difficult),
                    'image/object/bbox/truncated': int64_feature(truncated),
                }
            )
        )


        return example

    def add_to_record(self, file_name, tfrecord_writer):
        example = self.convert_to_example(file_name)
        tfrecord_writer.write(example.SerializeToString())

    def run(self, pic_names, output_dir, shuffling=False, size=300):
        if shuffling:
            random.seed(1314)
            random.shuffle(pic_names)

        total_num = len(pic_names)

        for start in range(0, total_num, size):
            tf_filename = '%s/%03d.tfrecord' % (output_dir, start // size)
            tf_recorder = tf.python_io.TFRecordWriter(tf_filename)
            print('=>' * (start * 5 // total_num) + '{:.0f}% Finished'.format(start / total_num * 100))
            for pic_idx in range(start, min(start + 300, total_num)):
                pic_name = pic_names[pic_idx]
                self.add_to_record(pic_name, tf_recorder)

        print('=>' * 5 + '{:.0f}% Finished'.format(100))


def convert_val():
    writer = PascalVocWriter()
    pic_names = open('/home/yifeng/DataSets/VOCdevkit/VOC2012/ImageSets/Main/val.txt').readlines()
    pic_names = [i.strip(' \n') for i in pic_names]
    writer.run(pic_names, output_dir='/home/yifeng/DataSets/VOCdevkit/VOC2012/tf_detects/tf_records/val')


def convert_train():
    writer = PascalVocWriter()
    pic_names = open('/home/yifeng/DataSets/VOCdevkit/VOC2012/ImageSets/Main/train.txt').readlines()
    pic_names = [i.strip(' \n') for i in pic_names]
    writer.run(pic_names, output_dir='/home/yifeng/DataSets/VOCdevkit/VOC2012/tf_detects/tf_records/train')


if __name__ == '__main__':
    # convert_val()
    convert_train()
