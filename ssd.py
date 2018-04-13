import numpy as np
import tensorflow as tf

from backbones.vgg_16 import vgg_conv_block
from datasets.pascal_voc_reader import get_dataset, get_next_batch, TRAIN_DIR
from tf_ops.wrap_ops import max_pool2d, conv2d, batch_norm2d, tensor_shape, drop_out,\
    softmax_with_logits, smooth_l1, l2_norm_1D, LOSS_COLLECTIONS

arg_scope = tf.contrib.framework.arg_scope
DEBUG_COLLECTIONS = 'debug_collections'


class SSDParameters(object):
    def __init__(self):
        self.img_shape = (300, 300)
        self.num_classes = 21
        self.no_annotation_label = 21
        self.feat_layers = ['block_4', 'block_7', 'block_8', 'block_9', 'block_10', 'block_11']
        self.feat_steps = [8, 16, 32, 64, 100, 300]
        self.anchor_scales = np.asarray([21, 45, 99, 153, 207, 261, 315]) / 300
        self.anchor_ratios = \
            [
                [1, 2, .5],
                [1, 2, .5, 3, 1. / 3],
                [1, 2, .5, 3, 1. / 3],
                [1, 2, .5, 3, 1. / 3],
                [1, 2, .5],
                [1, 2, .5]
            ]
        self.anchor_prior_scaling = [0.1, 0.1, 0.2, 0.2]
        self.f2b_l2_norm = [True, False, False, False, False, False, False]


default_params = SSDParameters()


def ssd_arg_scope(weight_init=None, weight_reg=None,
                  bias_init=tf.zeros_initializer, bias_reg=None, is_training=True):
    with arg_scope([conv2d],
                   batch_norm=False, use_bias=True, activate=tf.nn.relu,
                   weight_init=weight_init, weight_reg=weight_reg,
                   bias_init=bias_init, bias_reg=bias_reg, padding='SAME'):
        with arg_scope([batch_norm2d, drop_out], is_training=is_training):
            with arg_scope([max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def ssd_vgg16(inputs, scope=None):
    with tf.variable_scope(scope, 'ssd_vgg16', [inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope([conv2d, max_pool2d],
                       outputs_collections=end_points_collection):
            end_points = {}
            net = vgg_conv_block(inputs=inputs, outc=64, times=2, scope='conv1')
            end_points['block_1'] = net
            net = max_pool2d(net, ksize=[2, 2], name='pool1')

            net = vgg_conv_block(inputs=net, outc=128, times=2, scope='conv2')
            end_points['block_2'] = net
            net = max_pool2d(net, ksize=[2, 2], name='pool2')

            net = vgg_conv_block(inputs=net, outc=256, times=3, scope='conv3')
            end_points['block_3'] = net
            net = max_pool2d(net, ksize=[2, 2], name='pool3')

            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv4')
            end_points['block_4'] = net
            net = max_pool2d(net, ksize=[2, 2], name='pool4')

            net = vgg_conv_block(inputs=net, outc=512, times=3, scope='conv5')
            # here use different kernel size and stride to pool
            end_points['block_5'] = net
            net = max_pool2d(net, ksize=[3, 3], strides=[1, 1], name='pool5')

            # here starts ssd extra block
            net = conv2d(net, 1024, ksize=[3, 3], strides=[1, 1], ratios=[6, 6], name='conv6')
            end_points['block_6'] = net
            # net = drop_out(net, kp_prob=0.5)
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

            prediction_gathers = layers_predictions(end_points)

            return net, end_points, prediction_gathers


def _layer_prediction(feature_map, num_anchors, num_classes, l2_norm=False, name=None):
    """
    For each location in feature map, predict 4*num_anchors locations and num_classes objectness
    :param feature_map: [None, H, W, C]
    :param num_classes:
    :param name:
    :return: locations with shape [None, H, W, num_anchors, 4]
             scores with shape [None, H, W, num_anchors, num_classes]
    """
    with tf.variable_scope(name, 'feature2bbox'):
        if l2_norm:
            feature_map = l2_norm_1D(feature_map, scale=True)
        locations = conv2d(feature_map, outc=4 * num_anchors, ksize=[3, 3], activate=None, name='conv_loc')
        scores = conv2d(feature_map, outc=num_anchors * num_classes, ksize=[3, 3], activate=None, name='conv_cls')

        partial_shape = (tensor_shape(feature_map))[1:-1]

        locations = tf.reshape(locations, shape=[-1] + partial_shape + [num_anchors, 4])
        scores = tf.reshape(scores, shape=[-1] + partial_shape + [num_anchors, num_classes])

        # batch size = 1 version
        locations = tf.squeeze(locations, axis=0)
        scores = tf.squeeze(scores, axis=0)
        return locations, scores


def _layer_anchors(feature_shape, feature_step, scale_c, scale_n, ratios):
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
    y = (y + 0.5) * feature_step  / default_params.img_shape[0]
    x = (x + 0.5) * feature_step / default_params.img_shape[1]

    h = []
    w = []
    # use strange order here to match origin caffe order!
    h.append(scale_c / np.sqrt(ratios[0]))
    w.append(scale_c * np.sqrt(ratios[0]))

    # return one extra scale for aspect ratio = 1, as described in Original Paper
    h.append(np.sqrt(scale_c * scale_n))
    w.append(np.sqrt(scale_c * scale_n))

    for ratio in ratios[1:]:
        h.append(scale_c / np.sqrt(ratio))
        w.append(scale_c * np.sqrt(ratio))

    # for ratio in ratios:
    #     h.append(scale_c / np.sqrt(ratio))
    #     w.append(scale_c * np.sqrt(ratio))
    #
    # # return one extra scale for aspect ratio = 1, as described in Original Paper
    # h.append(np.sqrt(scale_c * scale_n))
    # w.append(np.sqrt(scale_c * scale_n))

    return y, x, np.asarray(h, np.float32), np.asarray(w, np.float32)


def _layer_encoding(layer_anchors, labels, bboxes, background_label=0):
    anchors_cy, anchors_cx, anchors_h, anchors_w = layer_anchors
    # support broadcasting
    anchors_ymin = anchors_cy - (anchors_h / 2.0)
    anchors_xmin = anchors_cx - (anchors_w / 2.0)
    anchors_ymax = anchors_cy + (anchors_h / 2.0)
    anchors_xmax = anchors_cx + (anchors_w / 2.0)

    # convert into four corners
    anchors_volume = anchors_h * anchors_w

    assert anchors_cy.shape == anchors_cx.shape and anchors_h.shape == anchors_w.shape

    # [fh, fW, num_anchors]
    anchors_shape = anchors_ymin.shape

    # for each anchor, assign a label for it
    # Steps:
    #   1. for each gt bbox, solve for jaccard idx of anchors with it
    #   2. assign each anchor
    encode_labels = background_label * tf.ones(anchors_shape, dtype=tf.int32)
    encode_ious = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_xmin = tf.zeros(anchors_shape, dtype=tf.float32)
    encode_ymax = tf.ones(anchors_shape, dtype=tf.float32)
    encode_xmax = tf.ones(anchors_shape, dtype=tf.float32)
    # remove the None(batch size) = 1
    bboxes = tf.squeeze(bboxes, axis=0)
    labels = tf.squeeze(labels, axis=0)

    def condition(idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax):
        return tf.less(idx, tf.shape(bboxes)[0])

    def body(idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax):
        # keep retrieve order the same as reading order in dataset!
        bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax = tf.unstack(bboxes[idx, :])
        bbox_volume = (bbox_ymax - bbox_ymin) * (bbox_xmax - bbox_xmin)

        inter_ymin = tf.maximum(bbox_ymin, anchors_ymin)
        inter_xmin = tf.maximum(bbox_xmin, anchors_xmin)
        inter_ymax = tf.minimum(bbox_ymax, anchors_ymax)
        inter_xmax = tf.minimum(bbox_xmax, anchors_xmax)

        inter_volume = tf.maximum(inter_ymax - inter_ymin, 0) * \
                       tf.maximum(inter_xmax - inter_xmin, 0)

        selector = tf.cast(tf.not_equal(inter_volume, 0), tf.float32)

        denominator = selector * (anchors_volume + bbox_volume - inter_volume) + (1 - selector)

        ious = inter_volume / (denominator)

        # update
        selector = tf.cast(ious > encode_ious, tf.int32)
        encode_labels = selector * labels[idx] + (1 - selector) * encode_labels
        selector = tf.cast(selector, tf.float32)
        encode_ymin = selector * bbox_ymin + (1 - selector) * encode_ymin
        encode_xmin = selector * bbox_xmin + (1 - selector) * encode_xmin
        encode_ymax = selector * bbox_ymax + (1 - selector) * encode_ymax
        encode_xmax = selector * bbox_xmax + (1 - selector) * encode_xmax

        encode_ious = tf.maximum(encode_ious, ious)
        return [idx + 1, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax]

    idx = 0
    [idx, bboxes, encode_labels, encode_ious, encode_ymin, encode_xmin, encode_ymax, encode_xmax] = \
        tf.while_loop(cond=condition, body=body, loop_vars=[idx, bboxes,
                                                            encode_labels,
                                                            encode_ious,
                                                            encode_ymin,
                                                            encode_xmin,
                                                            encode_ymax,
                                                            encode_xmax])

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
    # use SSD official orders instead !!!
    encode_locations = tf.stack([encode_cx, encode_cy, encode_w, encode_h], axis=-1)
    return encode_locations, encode_labels, encode_ious


def _layer_decode(locations, layer_anchors, clip=True):
    """
    Do bbox regression according to anchors positions and scales
    :param locations: [None, H, W, ,K ,4]
    :param layer_anchors: y[H, W, 1], x[H, W, 1], h[K], w[K]
    :param clip: whether clip the decode boxes in image
    :return: [H, W, K, 4]
    """
    anchors_y, anchors_x, anchors_h, anchors_w = layer_anchors
    # note the channel orders corresponding to anchors_encode there !
    # [H,W,K]
    pred_x, pred_y, pred_w, pred_h = tf.unstack(locations, axis=-1)

    pred_x = anchors_w * pred_x * default_params.anchor_prior_scaling[0] + anchors_x
    pred_y = anchors_h * pred_y * default_params.anchor_prior_scaling[1] + anchors_y
    pred_w = anchors_w * tf.exp(pred_w * default_params.anchor_prior_scaling[2])
    pred_h = anchors_h * tf.exp(pred_h * default_params.anchor_prior_scaling[3])

    pred_xmin = pred_x - pred_w / 2.0
    pred_ymin = pred_y - pred_h / 2.0
    pred_xmax = pred_x + pred_w / 2.0
    pred_ymax = pred_y + pred_h / 2.0

    if clip:
        pred_xmin = tf.maximum(pred_xmin, 0)
        pred_ymin = tf.maximum(pred_ymin, 0)
        pred_xmax = tf.minimum(pred_xmax, 1)
        pred_ymax = tf.minimum(pred_ymax, 1)

        # change channel orders again! use back to casual order [y1, x1, y2, x2] to use nms
    bboxes = tf.stack([pred_ymin, pred_xmin, pred_ymax, pred_xmax], axis=-1)
    return bboxes


def _layer_loss(locations, scores, encode_locations, encode_labels, encode_ious, pos_th, neg_ratio):
    """
    Calculate loss for one layer
    :param locations: predicted locations [1, H, W, K, 4 ]
    :param scores: predicted scores [1, H, W, K, 21]
    :param encode_locations: [H, W, K, 4]
    :param encode_labels: [H, W, K]
    :param encode_ious: [H, W, K]
    :return:
    """
    positive_mask = encode_ious > pos_th
    positive_num = tf.reduce_sum(tf.cast(positive_mask, tf.int32))
    # if no positive , ensure still some negatives, e.g. at least four bboxes along H side
    negative_num = tf.maximum(neg_ratio * positive_num, tf.shape(encode_locations)[0] * 4)
    # ensure it is less than the number of all available bboxes
    negative_num = tf.minimum(negative_num, tf.reduce_prod(encode_ious.shape))
    # Hard Negative Mining
    neg_values, _ = tf.nn.top_k(tf.reshape(-1.0 * encode_ious, [-1]), k=negative_num)
    # in case that -neg_value[-1] is larger than pos_th
    negative_mask = tf.logical_and(
        encode_ious < -neg_values[-1],
        tf.logical_not(positive_mask)
    )

    positive_mask = tf.cast(positive_mask, tf.float32)
    negative_mask = tf.cast(negative_mask, tf.float32)
    tf.add_to_collection(DEBUG_COLLECTIONS, -neg_values[-1])
    with tf.name_scope('cross_entropy_loss'):
        with tf.name_scope('positive'):
            pos_loss = softmax_with_logits(predictions=scores,
                                           labels=encode_labels,
                                           ignore_labels=[],
                                           weights=tf.reshape(positive_mask, [-1]),
                                           loss_collections=LOSS_COLLECTIONS)
        with tf.name_scope('negative'):
            neg_loss = softmax_with_logits(predictions=scores,
                                           labels=encode_labels,
                                           ignore_labels=[],
                                           weights=tf.reshape(negative_mask, [-1]),
                                           loss_collections=LOSS_COLLECTIONS
                                           )

    with tf.name_scope('bbox_regression_loss'):
        reg_loss = smooth_l1(locations - encode_locations)
        reg_loss = tf.reduce_sum(reg_loss, axis=-1)

        # [H*W*K]
        reg_loss = tf.reduce_mean(
            reg_loss * positive_mask,
            name='regression_loss'
        )
        tf.add_to_collection(LOSS_COLLECTIONS, reg_loss)

    return pos_loss, neg_loss, reg_loss


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
        locations, scores = _layer_prediction(layer, num_anchors=1 + num_ratios,
                                              num_classes=default_params.num_classes,
                                              l2_norm=default_params.f2b_l2_norm[idx],
                                              name='feature2bbox{}'.format(idx + 1))
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
                                    default_params.feat_steps[idx],
                                    default_params.anchor_scales[idx],
                                    default_params.anchor_scales[idx + 1],
                                    default_params.anchor_ratios[idx])
        ys.append(y)
        xs.append(x)
        hs.append(h)
        ws.append(w)
    return ys, xs, hs, ws


def layers_encoding(all_anchors, labels, bboxes):
    gather_locations, gather_labels, gather_ious = [], [], []
    ys, xs, hs, ws = all_anchors
    for idx in range(len(ys)):
        anchor = ys[idx], xs[idx], hs[idx], ws[idx]
        encode_locations, encode_labels, encode_ious = \
            _layer_encoding(anchor, labels, bboxes)
        gather_locations.append(encode_locations)
        gather_labels.append(encode_labels)
        gather_ious.append(encode_ious)
    return gather_locations, gather_labels, gather_ious


def layers_decoding(gather_locations, gather_anchors, clip=True):
    ys, xs, hs, ws = gather_anchors
    gather_decode_bboxes = []
    for idx in range(len(ys)):
        tmp = ys[idx], xs[idx], hs[idx], ws[idx]
        decode_bboxes = _layer_decode(gather_locations[idx], tmp, clip=clip)
        gather_decode_bboxes.append(decode_bboxes)
    return gather_decode_bboxes


def layers_loss(prediction_gathers, encoding_gathers, pos_th=0.5, neg_ratio=3):
    gather_pred_locations, gather_pred_scores = prediction_gathers
    gather_truth_locations, gather_truth_labels, gather_truth_ious = encoding_gathers

    gather_pos_loss, gather_neg_loss, gather_reg_loss = [], [], []
    for idx in range(len(default_params.feat_layers)):
        pos_loss, neg_loss, reg_loss = _layer_loss(
            locations=gather_pred_locations[idx],
            scores=gather_pred_scores[idx],
            encode_locations=gather_truth_locations[idx],
            encode_labels=gather_truth_labels[idx],
            encode_ious=gather_truth_ious[idx],
            pos_th=pos_th,
            neg_ratio=neg_ratio
        )
        gather_pos_loss.append(pos_loss)
        gather_neg_loss.append(neg_loss)
        gather_reg_loss.append(reg_loss)
    return gather_pos_loss, gather_neg_loss, gather_reg_loss


def layers_select_nms(gather_pred_scores, gather_decode_bboxes, select_th=0.5, nms_th=0.45, nms_k=200, num_classes=21):
    all_scores = []
    all_bboxes = []

    for idx in range(len(gather_pred_scores)):
        scores = gather_pred_scores[idx]
        # softmax here to put all comparisons in the same scale
        scores = tf.reshape(tf.nn.softmax(scores, axis=-1), [-1, 21])
        decode_bboxes = tf.reshape(gather_decode_bboxes[idx], [-1, 4])

        all_scores.append(scores)
        all_bboxes.append(decode_bboxes)

    # [N, 21]
    scores = tf.concat(all_scores, axis=0)
    # [N, 4]
    bboxes = tf.concat(all_bboxes, axis=0)

    gather_scores = {}
    gather_bboxes = {}
    class_scores_list = tf.unstack(scores, axis=-1)

    for class_id in range(num_classes):
        class_score = class_scores_list[class_id]
        # set all values below th to be 0
        # selector = tf.cast(class_score >= select_th, tf.float32)
        # select_score = class_score * selector
        # select_boxes = bboxes * tf.expand_dims(selector, axis=-1)

        # use nms to filter
        nms_idxes = tf.image.non_max_suppression(bboxes, class_score,
                                                 max_output_size=nms_k,
                                                 iou_threshold=nms_th)
        nms_score = tf.gather(class_score, nms_idxes)
        nms_boxes = tf.gather(bboxes, nms_idxes)

        select_idxes = tf.where(nms_score >= select_th)
        select_score = tf.gather(nms_score, select_idxes)
        select_boxes = tf.gather(nms_boxes, select_idxes)


        gather_scores[class_id] = select_score
        gather_bboxes[class_id] = select_boxes

    return gather_scores, gather_bboxes


if __name__ == '__main__':
    dataset = get_dataset(dir=TRAIN_DIR, batch_size=1, num_epochs=1)
    name, image, labels, bboxes = get_next_batch(dataset)

    # name_v, image_v, labels_v, bboxes_v = sess.run([name, image, labels, bboxes])
    # print(name_v)
    # print(bboxes_v)

    # inputs = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32, name='inputs')
    with arg_scope(ssd_arg_scope()):
        net, end_points, prediction_gathers = ssd_vgg16(image, scope='ssd_vgg16_300')
        gather_locations, gather_scores = prediction_gathers

    gather_anchors = layers_anchors(end_points)
    gather_decode_bboxes = layers_decoding(gather_locations, gather_anchors)
    gather_scores, gather_bboxes = layers_select_nms(gather_scores, gather_decode_bboxes)
    # # For training
    # encoding_gathers = layers_encoding(all_anchors, labels, bboxes)
    # gather_pos_loss, gather_neg_loss, gather_reg_loss = layers_loss(prediction_gathers, encoding_gathers)
    #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([gather_locations])

    # For Evaling
    # print(labels_v)
