import os

from datasets.pascal_voc_reader import get_dataset, get_next_batch, TRAIN_DIR
from datasets.pascal_voc_utils import pascal_voc_classes
from ssd import ssd_arg_scope, ssd_vgg16, default_params, layers_anchors, \
    layers_loss, layers_encoding, layers_decoding, layers_select_nms
from tf_ops.benchmarks import mAP, mIOU
from tf_ops.visualize import draw_bbox
from tf_ops.wrap_ops import *
from tf_utils import partial_restore, add_gradient_summary, \
    add_var_summary, add_activation_summary, parse_device_name, add_iou_summary

arg_scope = tf.contrib.framework.arg_scope

LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

# pre settings
tf.app.flags.DEFINE_string('data_dir', TRAIN_DIR, 'where training set is put')
tf.app.flags.DEFINE_integer('reshape_height', 300, 'reshape height')
tf.app.flags.DEFINE_integer('reshape_weight', 300, 'reshape weight')
tf.app.flags.DEFINE_integer('num_classes', 21, '#classes')

# learning configs
tf.app.flags.DEFINE_integer('epoch_num', 1, 'epoch_nums')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.app.flags.DEFINE_float('weight_learning_rate', 1e-3, 'weight learning rate')
tf.app.flags.DEFINE_float('bias_learning_rate', None, 'bias learning rate')
tf.app.flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')
tf.app.flags.DEFINE_float('learning_decay', 0.99, 'learning rate decay')
tf.app.flags.DEFINE_float('momentum', 0.99, 'momentum')

# deploy configs
tf.app.flags.DEFINE_string('store_device', '/CPU:0', 'where to place the variables')
tf.app.flags.DEFINE_string('run_device', '/CPU:0', 'where to run the models')
tf.app.flags.DEFINE_float('gpu_fraction', 0.8, 'gpu memory fraction')
tf.app.flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
tf.app.flags.DEFINE_float('weight_reg_scale', 4e-5, 'weight regularization scale')
tf.app.flags.DEFINE_string('weight_reg_func', 'l2', 'use which func to regularize weight')
tf.app.flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
tf.app.flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# # model load & save configs
# tf.app.flags.DEFINE_string('summaries_dir', '/home/chenyifeng/TF_Logs/SSD/vgg16_300/sgpu',
#                            'where to store summary log')
#
# tf.app.flags.DEFINE_string('pretrained_ckpts', '/home/chenyifeng/TF_Models/ptrain/ssd_vgg16_300/SSD_VGG300_120000.ckpt',
#                            'where to load pretrained model')
#
# tf.app.flags.DEFINE_string('last_ckpt', '/home/chenyifeng/TF_Models/atrain/SSD/vgg16_300/sgpu',
#                            'where to load last saved model')
#
# tf.app.flags.DEFINE_string('next_ckpt', '/home/chenyifeng/TF_Models/atrain/SSD/vgg16_300/sgpu',
#                            'where to store current model')

# CPU VERSION
# model load & save configs
tf.app.flags.DEFINE_string('summaries_dir', '/home/yifeng/TF_Logs/SSD/vgg16_300/sgpu',
                           'where to store summary log')

tf.app.flags.DEFINE_string('pretrained_ckpts', '/home/yifeng/TF_Models/ckpts/ssd_vgg16_300/SSD_VGG300_120000.ckpt',
                           'where to load pretrained model')

tf.app.flags.DEFINE_string('last_ckpt', None,
                           'where to load last saved model')

tf.app.flags.DEFINE_string('next_ckpt', '/home/yifeng/TF_Models/atrain/SSD/vgg16_300/sgpu',
                           'where to store current model')


tf.app.flags.DEFINE_integer('save_per_step', 5817, 'save model per xxx steps')

FLAGS = tf.app.flags.FLAGS

if (FLAGS.reshape_height is None or FLAGS.reshape_weight is None) and FLAGS.batch_size != 1:
    assert 0, 'Can''t Stack Images Of Different Shapes, Please Speicify Reshape Size!'

store_device = parse_device_name(FLAGS.store_device)
run_device = parse_device_name(FLAGS.run_device)
weight_reg = regularizer(mode=FLAGS.weight_reg_func, scale=FLAGS.weight_reg_scale)
bias_reg = regularizer(mode=FLAGS.bias_reg_func, scale=FLAGS.bias_reg_scale)

# config devices
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
if FLAGS.run_device in '01234567':
    print('Deploying Model on {} GPU Card'.format(''.join(FLAGS.run_device)))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
else:
    print('Deploying Model on CPU')

# set up step
sess = tf.Session(config=config)

default_params.num_classes = FLAGS.num_classes
with tf.device(FLAGS.run_device):
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    # read data
    default_params.img_shape = [FLAGS.reshape_height, FLAGS.reshape_weight]
    name_batch, image_batch, labels_batch, bboxes_batch  = get_next_batch(get_dataset(
        dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, num_epochs=FLAGS.epoch_num, reshape_size=default_params.img_shape)
    )

# inference
with arg_scope([get_variable], device=store_device):
    with tf.device('/CPU:0'):
        with arg_scope(ssd_arg_scope(weight_init=None, weight_reg=weight_reg,
                  bias_init=tf.zeros_initializer, bias_reg=bias_reg, is_training=False)):
            net, endpoints, prediction_gathers = ssd_vgg16(image_batch, scope='ssd_vgg16_300')

        # predictions of bboxes
        gather_pred_locations, gather_scores = prediction_gathers

        # encode ground truth for bboxes
        gather_anchors = layers_anchors(endpoints)
        encoding_gathers = layers_encoding(gather_anchors, labels_batch, bboxes_batch)
        # gather_gt_locations, gather_labels, gather_ious = encoding_gathers

        # solve for loss
        gather_pos_loss, gather_neg_loss, gather_box_loss = layers_loss(prediction_gathers, encoding_gathers)

        # for visualization
        gather_decode_bboxes = layers_decoding(gather_pred_locations, gather_anchors, clip=True)
        scores, bboxes = layers_select_nms(gather_scores, gather_decode_bboxes, select_th=0.5)
        image_batch = draw_bbox(image_batch, scores, bboxes)

        # TODO : ? check other strategies!
        pos_loss = tf.reduce_sum(gather_pos_loss)
        neg_loss = tf.reduce_sum(gather_neg_loss)
        box_loss = tf.reduce_sum(gather_box_loss)

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.add_n(reg_loss)

        # total_loss = pos_loss + neg_loss + box_loss + reg_loss
        total_loss = pos_loss + neg_loss + box_loss

        # solve for gradients
        weight_vars = tf.get_collection(weight_collections)
        bias_vars = tf.get_collection(bias_collections)


    # add summaries
    with tf.name_scope('summary_input_output'):
        tf.summary.image('image_batch', image_batch, max_outputs=1)
        tf.summary.scalar('pos_loss', pos_loss)
        tf.summary.scalar('neg_loss', neg_loss)
        tf.summary.scalar('box_loss', box_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', total_loss)
        # tf.summary.scalar('learning_rate', decay_learning_rate)


    with tf.name_scope('summary_vars'):
        for weight in weight_vars:
            add_var_summary(weight)
        for bias in bias_vars:
            add_var_summary(bias)

    with tf.name_scope('summary_activations'):
        for activations in endpoints.keys():
            add_activation_summary(endpoints[activations])

merge_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=3)

sess.run(tf.global_variables_initializer())

# initialize
ckpt = None
if FLAGS.last_ckpt is not None:
    ckpt = tf.train.latest_checkpoint(FLAGS.last_ckpt)
    if ckpt is not None:
        # set up save configuration
        saver.restore(sess, ckpt)
        print('Recovering From {}'.format(ckpt))
    else:
        print('No previous Model Found in {}'.format(ckpt))
elif FLAGS.pretrained_ckpts is not None:
    # pre-train priority higher
    partial_restore_op = partial_restore(sess, tf.global_variables(), FLAGS.pretrained_ckpts)
    sess.run(partial_restore_op)
    print('Recovering From Pretrained Model {}'.format(FLAGS.pretrained_ckpts))


try:
    # start training
    local_step = 0
    sess.run(tf.local_variables_initializer())
    while True:  # train until OutOfRangeError
        pos_loss_v, neg_loss_v, box_loss_v, reg_loss_v, total_loss_v, step, summary = \
            sess.run([pos_loss, neg_loss, box_loss, reg_loss, total_loss, global_step, merge_summary])
        train_writer.add_summary(summary, step)
        local_step += 1

        print("Step {} : Total Loss {:.3f}  Pos Loss {:.3f}  Neg loss {:.3f} Box Loss {:.3f} REG Loss {:.3f}"
              .format(step, total_loss_v, pos_loss_v, neg_loss_v, box_loss_v, reg_loss_v))

except tf.errors.OutOfRangeError:
    print('Done Inferencing')
