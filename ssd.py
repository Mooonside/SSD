from collections import namedtuple

import numpy as np
import tensorflow as tf

from backbones.vgg_16 import vgg_conv_block
from tf_ops.wrap_ops import max_pool2d, conv2d, batch_norm2d, tensor_shape

arg_scope = tf.contrib.framework.arg_scope

# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'anchor_scales',
                                         'anchor_ratios',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

default_params = SSDParams(
    img_shape=(300, 300),
    num_classes=21,
    no_annotation_label=21,
    feat_layers=['block_4', 'block_7', 'block_8', 'block_9', 'block_10', 'block_11'],
    anchor_scales=np.linspace(0.20, 0.90, 6 + 1),
    anchor_ratios=[[2, .5],
                   [2, .5, 3, 1. / 3],
                   [2, .5, 3, 1. / 3],
                   [2, .5, 3, 1. / 3],
                   [2, .5],
                   [2, .5]],
    normalizations=[20, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2]
)


def ssd_arg_scope(weight_init=None, weight_reg=None,
                  bias_init=tf.zeros_initializer, bias_reg=None, is_training=True):
    with arg_scope([conv2d],
                   batch_norm=False, activate=tf.nn.relu,
                   weight_init=weight_init, weight_reg=weight_reg,
                   bias_init=bias_init, bias_reg=bias_reg, padding='SAME'):
        with arg_scope([batch_norm2d], is_training=is_training) as arg_sc:
            return arg_sc


def ssd_vgg16(inputs, scope='ssd_vgg16'):
    with tf.variable_scope(scope, 'ssd_vgg16', [inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope([conv2d, max_pool2d],
                       outputs_collections=end_points_collection):
            end_points = {}
            net = vgg_conv_block(inputs=inputs, outc=64, times=2, scope='conv1')
            net = max_pool2d(net, ksize=[2, 2], name='pool1')
            end_points['block_1'] = net

            net = vgg_conv_block(inputs=net, outc=128, times=2, scope='conv2')
            net = max_pool2d(net, ksize=[2, 2], name='pool2')
            end_points['block_2'] = net

            net = vgg_conv_block(inputs=net, outc=256, times=3, scope='conv3')
            net = max_pool2d(net, ksize=[2, 2], name='pool3')
            end_points['block_3'] = net

            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv4')
            net = max_pool2d(net, ksize=[2, 2], name='pool4')
            end_points['block_4'] = net

            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv5')
            # here use different kernel size and stride to pool
            net = max_pool2d(net, ksize=[3, 3], strides=[1, 1], name='pool5')
            end_points['block_5'] = net

            # here starts ssd extra block
            net = conv2d(net, 1024, ksize=[3, 3], strides=[1, 1], ratios=[6, 6], name='conv6')
            end_points['block_6'] = net
            net = conv2d(net, 1024, ksize=[1, 1], strides=[1, 1], ratios=[1, 1], name='conv7')
            end_points['block_7'] = net

            with tf.variable_scope('conv8'):
                net = conv2d(net, 256, ksize=[1, 1], strides=[1, 1], name='conv8_1')
                net = conv2d(net, 512, ksize=[3, 3], strides=[2, 2], name='conv8_2')
            end_points['block_8'] = net

            with tf.variable_scope('conv9'):
                net = conv2d(net, 128, ksize=[1, 1], strides=[1, 1], name='conv9_1')
                net = conv2d(net, 256, ksize=[3, 3], strides=[2, 2], name='conv9_2')
            end_points['block_9'] = net

            with tf.variable_scope('conv10'):
                net = conv2d(net, 128, ksize=[1, 1], strides=[1, 1], name='conv10_1')
                net = conv2d(net, 256, ksize=[3, 3], strides=[1, 1], padding='VALID', name='conv10_2')
            end_points['block_10'] = net

            with tf.variable_scope('conv11'):
                net = conv2d(net, 128, ksize=[1, 1], strides=[1, 1], name='conv11_1')
                net = conv2d(net, 256, ksize=[3, 3], strides=[1, 1], padding='VALID', name='conv11_2')
            end_points['block_11'] = net

            [end_points.update({i.name: i}) for i in tf.get_collection(end_points_collection)]

            return net, end_points


def _layer_prediction(feature_map, num_anchors, num_classes, name=None):
    """
    For each location in feature map, predict 4*num_anchors locations and num_classes objectness
    :param feature_map: [None, H, W, C]
    :param num_classes:
    :param name:
    :return: locations with shape [None, H, W, num_anchors, 4]
             scores with shape [None, H, W, num_anchors, num_classes]
    """
    with tf.variable_scope(name, 'feature2bbox'):
        locations = conv2d(feature_map, outc=4 * num_anchors, ksize=[3, 3], activate=None, name='conv_loc')
        scores = conv2d(feature_map, outc=num_anchors * num_classes, ksize=[3, 3], activate=None, name='conv_cls')

        partial_shape = (tensor_shape(feature_map))[1:-1]

        locations = tf.reshape(locations, shape=[-1] + partial_shape + [num_anchors, 4])
        scores = tf.reshape(scores, shape=[-1] + partial_shape + [num_anchors, num_classes])
        return locations, scores


def _layer_anchors(feature_shape, scale_c, scale_n, ratios):
    """
    For feature shape, encode them into scale bboxes with different ratios
    :param feature_shape: [H ,W]
    :param scale_c: current layer bbox scale
    :param scale_n: next layer bbox scale
    :param ratios: different aspect ratios
    :return: y, x, h ,w
    """
    # x in [ (0...fw), ..., (0....fw)]
    # y in [ (0...0), ..., (fh...fh)]
    y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
    # support broadcasting in encoding part
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)
    # relative position in feature map is the aligned relative position in whole image
    y = (y + 0.5) / feature_shape[0]
    x = (x + 0.5) / feature_shape[1]

    h = []
    w = []
    for ratio in ratios:
        h.append(scale_c / np.sqrt(ratio))
        w.append(scale_c * np.sqrt(ratio))

    # return one extra scale for aspect ratio = 1, as described in Original Paper
    h.append(np.sqrt(scale_c * scale_n))
    w.append(np.sqrt(scale_c * scale_n))
    return y, x, h, w


def _layer_encoding(layer_anchors, labels, bboxes, ignore_label=default_params.num_classes + 1):
    anchors_cy, anchors_cx, anchors_h, anchors_w = layer_anchors
    # support broadcasting
    anchors_ymin = anchors_cy - (anchors_h / 2.0)
    anchors_xmin = anchors_cx - (anchors_w / 2.0)
    anchors_ymax = anchors_cy + (anchors_h / 2.0)
    anchors_xmax = anchors_cx + (anchors_w / 2.0)

    # convert into four corners
    anchors_volume = anchors_h * anchors_w

    assert anchors_cy.shape == anchors_cy.shape and anchors_h.shape == anchors_w.shape

    # [fh, fW, num_anchors]
    anchors_shape = anchors_cy.shape + anchors_h.shape

    # for each anchor, assign a label for it
    # Steps:
    #   1. for each gt bbox, solve for jaccard idx of anchors with it
    #   2. assign each anchor
    encode_labels = ignore_label * tf.ones(anchors_shape, dtype=tf.int32)
    encode_ious = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_xmin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymax = tf.ones(anchors_shape, dtype=tf.float32)
    encode_xmax = tf.ones(anchors_shape, dtype=tf.float32)

    for idx, bbox in enumerate(bboxes):
        bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax = bbox
        bbox_volume = (bbox_ymax - bbox_ymin) * (bbox_xmax - bbox_xmin)

        inter_ymin = tf.maximum(bbox_ymin, anchors_ymin)
        inter_xmin = tf.maximum(bbox_xmin, anchors_xmin)
        inter_ymax = tf.minimum(bbox_ymax, anchors_ymax)
        inter_xmax = tf.minimum(bbox_xmax, anchors_xmax)

        inter_volume = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        denominator = anchors_volume + bbox_volume - inter_volume
        ious = inter_volume / denominator

        # update
        selector = tf.cast(ious > encode_ious, tf.int32)
        encode_labels = selector * labels[idx] + (1 - selector) * encode_labels
        selector = tf.cast(selector, tf.float32)
        encode_ymin += selector * bbox_ymin + (1 - selector) * encode_ymin
        encode_xmin += selector * bbox_xmin + (1 - selector) * encode_xmin
        encode_ymax += selector * bbox_ymax + (1 - selector) * encode_ymax
        encode_xmax += selector * bbox_xmax + (1 - selector) * encode_xmax

        encode_ious = max(encode_ious, ious)
    # reform to center, size pattern
    encode_cy = (encode_ymin + encode_ymax) / 2
    encode_cx = (encode_xmin + encode_xmax) / 2
    encode_h = encode_ymax - encode_ymin
    encode_w = encode_xmax - encode_xmin
    # Do Bbox regression here
    # [h , w, c]  = ([h , w, c] - [h , w, 1]) / [c]
    encode_cy = (encode_cy - anchors_cy) / anchors_h
    encode_cx = (encode_cx - anchors_cx) / anchors_w
    encode_h = tf.log(encode_h / anchors_h)
    encode_w = tf.log(encode_w / anchors_w)

    encode_truth = tf.concat([encode_cy, encode_cx, encode_h, encode_w], axis=-1)
    return encode_truth, encode_labels, encode_ious


def layers_predictions(end_points):
    """
    Gather predictions from layers
    :param end_points:
    :return:
    """
    gather_locations, gather_scores = [], []
    for idx, key in enumerate(default_params.feat_layers):
        layer = end_points[key]
        num_ratios = len(default_params.anchor_ratios[idx])
        locations, scores = _layer_prediction(layer, num_anchors=2 + num_ratios,
                                              num_classes=default_params.num_classes)
        gather_locations.append(locations)
        gather_scores.append(scores)
    return gather_locations, gather_scores


def layers_anchors(end_points):
    """
    Gather anchors from layers
    :param end_points:
    :return:
    """
    ys, xs, hs, ws = [], [], [], []
    for idx, key in enumerate(default_params.feat_layers):
        layer = end_points[key]
        y, x, h, w = _layer_anchors(tensor_shape(layer)[1:-1],
                                    default_params.anchor_scales[idx],
                                    default_params.anchor_scales[idx + 1],
                                    default_params.anchor_ratios[idx])
        ys.append(y)
        xs.append(x)
        hs.append(h)
        ws.append(w)
    return ys, xs, hs, ws


def layers_encoding(all_anchors, labels, bboxes):
    gather_truth, gather_labels, gather_ious = [], [], []
    for anchor in all_anchors:
        encode_truth, encode_labels, encode_ious = \
            _layer_encoding(anchor, labels, bboxes)
        gather_truth.append(encode_truth)
        gather_labels.append(encode_labels)
        gather_ious.append(encode_ious)
    return gather_truth, gather_labels, gather_ious




# if __name__ == '__main__':
#     inputs = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32, name='inputs')
#     with arg_scope(ssd_arg_scope()):
#         net, end_points = ssd_vgg16(inputs)
#
#     layers_locations, layers_scores = layers_predictions(end_points)
#     print(layers_locations)
# layers_anchors_y, layers_anchors_x, layers_anchors_h, layers_anchors_w = layers_anchors(end_points)
