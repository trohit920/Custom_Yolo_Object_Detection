# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

import random
import numpy as np
import cv2
from collections import Counter

slim = tf.contrib.slim

import new_args as args

from tensorflow.core.framework import summary_pb2

# ****************************** layers ********************************************

def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net
    
    # first two conv2d layers
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

# ***************************************** Model Def ********************************************

class yolov3(object):

    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999, weight_decay=5e-4, use_static_shape=True):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
                         # [30, 61], [62, 45], [59,  119],
                         # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        # inference speed optimization
        # if `use_static_shape` is True, use tensor.get_shape(), otherwise use tf.shape(tensor)
        # static_shape is slightly faster
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map 
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        
        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''
        
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))
        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss
    

    def box_iou(self, pred_boxes, valid_true_boxes):
        '''
        param:
            pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
            valid_true: [V, 4]
        '''

        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    
    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

# ********************************* Utils *******************************************



def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)



def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def config_learning_rate(new_args, global_step):
    if new_args.lr_type == 'exponential':
        lr_tmp = tf.train.exponential_decay(new_args.learning_rate_init, global_step, new_args.lr_decay_freq,
                                            new_args.lr_decay_factor, staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, new_args.lr_lower_bound)
    elif new_args.lr_type == 'cosine_decay':
        train_steps = (new_args.total_epoches - float(new_args.use_warm_up) * new_args.warm_up_epoch) * new_args.train_batch_num
        return new_args.lr_lower_bound + 0.5 * (new_args.learning_rate_init - new_args.lr_lower_bound) * \
            (1 + tf.cos(global_step / train_steps * np.pi))
    elif new_args.lr_type == 'cosine_decay_restart':
        return tf.train.cosine_decay_restarts(new_args.learning_rate_init, global_step, 
                                              new_args.lr_decay_freq, t_mul=2.0, m_mul=1.0, 
                                              name='cosine_decay_learning_rate_restart')
    elif new_args.lr_type == 'fixed':
        return tf.convert_to_tensor(new_args.learning_rate_init, name='fixed_learning_rate')
    elif new_args.lr_type == 'piecewise':
        return tf.train.piecewise_constant(global_step, boundaries=new_args.pw_boundaries, values=new_args.pw_values,
                                           name='piecewise_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')

# ******************************************* Data augmentation *********************************************

def mix_up(img1, img2, bbox1, bbox2):
    '''
    return:
        mix_img: HWC format mix up image
        mix_bbox: [N, 5] shape mix up bbox, i.e. `x_min, y_min, x_max, y_mix, mixup_weight`.
    '''
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    mix_img = np.zeros(shape=(height, width, 3), dtype='float32')

    # rand_num = np.random.random()
    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))
    mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * rand_num
    mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - rand_num)

    mix_img = mix_img.astype('uint8')

    # the last element of the 2nd dimention is the mix up weight
    bbox1 = np.concatenate((bbox1, np.full(shape=(bbox1.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox2 = np.concatenate((bbox2, np.full(shape=(bbox2.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox = np.concatenate((bbox1, bbox2), axis=0)

    return mix_img, mix_bbox


def bbox_crop(bbox, crop_box=None, allow_outside_center=True):

    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox

def bbox_iou(bbox_a, bbox_b, offset=0):
    
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)


def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def resize_with_bbox(img, bbox, new_width, new_height, interp=0, letterbox=False):
    '''
    Resize the image and correct the bbox accordingly.
    '''

    if letterbox:
        image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox
    else:
        ori_height, ori_width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox


def random_flip(img, bbox, px=0, py=0):
    
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox


def random_expand(img, bbox, max_ratio=4, fill=0, keep_ratio=True):
    
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=img.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = img

    # correct bbox
    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox


# ************************************** Data utils ***************************************

iter_cnt = 0


def parse_line(line):
    
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    params:
        boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
        labels: [N] shape, int64 dtype.
        class_num: int64 num.
        anchors: [9, 4] shape, float32 dtype.
    '''
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight. 
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
    # [N]
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print(feature_map_group, '|', y,x,k,c)

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode, letterbox_resize):
    
    if not isinstance(line, list):
        img_idx, pic_path, boxes, labels, _, _ = parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    else:
        # the mix up case
        _, pic_path1, boxes1, labels1, _, _ = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, boxes2, labels2, _, _ = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, boxes = mix_up(img1, img2, boxes1, boxes2)
        labels = np.concatenate((labels1, labels2))

    if mode == 'train':
        # random color jittering
        # NOTE: applying color distort may lead to bad performance sometimes
        img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 4)

        # random cropping
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=interp, letterbox=letterbox_resize)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1, letterbox=letterbox_resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(batch_line, class_num, img_size, anchors, mode, multi_scale=False, mix_up=False, letterbox_resize=True, interval=10):
    
    global iter_cnt
    # multi_scale training
    if multi_scale and mode == 'train':
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # mix up strategy
    if mix_up and mode == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            if np.random.uniform(0, 1) < 0.5:
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines

    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(line, class_num, img_size, anchors, mode, letterbox_resize)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch

# ******************************************** Eval utils ****************************************

def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou



def evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, iou_thresh=0.5, calc_now=True):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    This function will perform gpu operation on the GPU.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 80] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels, each from 0 to 79
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = sess.run(gpu_nms_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, image_ids, y_pred):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    image_id = image_ids[0]

    # keep the first dimension 1
    pred_boxes = y_pred[0][0:1]
    pred_confs = y_pred[1][0:1]
    pred_probs = y_pred[2][0:1]

    boxes, scores, labels = sess.run(gpu_nms_op,
                                     feed_dict={pred_boxes_flag: pred_boxes,
                                                pred_scores_flag: pred_confs * pred_probs})

    pred_content = []
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max = boxes[i]
        score = scores[i]
        label = labels[i]
        pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label])

    return pred_content


gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        x_max * resize_ratio + dw,
                                        y_max * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([x_min * new_width / ori_width,
                                        y_min * new_height / ori_height,
                                        x_max * new_width / ori_width,
                                        y_max * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0, 0, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap

# ***************************************************************************

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]



###############################################################################

################################### training  ###########################################

def train_fn():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')

    # setting placeholders
    is_training = tf.placeholder(tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
    # register the gpu nms operation here for the following evaluation scheme
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

    ##################
    # tf.data pipeline
    ##################
    train_dataset = tf.data.TextLineDataset(args.train_file)
    train_dataset = train_dataset.shuffle(args.train_img_cnt)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                            inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
                            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)

    val_dataset = tf.data.TextLineDataset(args.val_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                            inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False, args.letterbox_resize],
                            Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    val_dataset.prefetch(args.prefetech_buffer)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # get an element from the chosen dataset iterator
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # tf.data pipeline will lose the data `static` shape, so we need to set it manually
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    ##################
    # Model definition
    ##################
    yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true)
    y_pred = yolo_model.predict(pred_feature_maps)

    l2_loss = tf.losses.get_regularization_loss()

    # setting restore parts and vars to update
    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
    update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

    tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
    tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
    tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
    tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
    tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
    tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

    global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    if args.use_warm_up:
        learning_rate = tf.cond(tf.less(global_step, args.train_batch_num * args.warm_up_epoch), 
                                lambda: args.learning_rate_init * global_step / (args.train_batch_num * args.warm_up_epoch),
                                lambda: config_learning_rate(args, global_step - args.train_batch_num * args.warm_up_epoch))
    else:
        learning_rate = config_learning_rate(args, global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    if not args.save_optimizer:
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    optimizer = config_optimizer(args.optimizer_name, learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
        # apply gradient clip to avoid gradient exploding
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        clip_grad_var = [gv if gv[0] is None else [
            tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    if args.save_optimizer:
        print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver_to_restore.restore(sess, args.restore_path)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        print('\n----------- start to train -----------\n')

        best_mAP = -np.Inf

        for epoch in range(args.total_epoches):

            sess.run(train_init_op)
            loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            for i in trange(args.train_batch_num):
                _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                    [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True})

                writer.add_summary(summary, global_step=__global_step)

                loss_total.update(__loss[0], len(__y_pred[0]))
                loss_xy.update(__loss[1], len(__y_pred[0]))
                loss_wh.update(__loss[2], len(__y_pred[0]))
                loss_conf.update(__loss[3], len(__y_pred[0]))
                loss_class.update(__loss[4], len(__y_pred[0]))

                if __global_step % args.train_evaluation_step == 0 and __global_step > 0:
                    # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)
                    recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred, __y_true, args.class_num, args.nms_threshold)

                    info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                            epoch, int(__global_step), loss_total.average, loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
                    info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, __lr)
                    print(info)
                    logging.info(info)

                    writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=__global_step)
                    writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=__global_step)

                    if np.isnan(loss_total.average):
                        print('****' * 10)
                        raise ArithmeticError(
                            'Gradient exploded! Please train again and you may need modify some parameters.')

            # NOTE: this is just demo. You can set the conditions when to save the weights.
            if epoch % args.save_epoch == 0 and epoch > 0:
                if loss_total.average <= 2.:
                    saver_to_save.save(sess, args.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(epoch, int(__global_step), loss_total.average, __lr))

            # switch to validation dataset for evaluation
            if epoch % args.val_evaluation_epoch == 0 and epoch >= args.warm_up_epoch:
                sess.run(val_init_op)

                val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                    AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                val_preds = []

                for j in trange(args.val_img_cnt):
                    __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss],
                                                            feed_dict={is_training: False})
                    pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)
                    val_preds.extend(pred_content)
                    val_loss_total.update(__loss[0])
                    val_loss_xy.update(__loss[1])
                    val_loss_wh.update(__loss[2])
                    val_loss_conf.update(__loss[3])
                    val_loss_class.update(__loss[4])

                # calc mAP
                rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)

                info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

                for ii in range(args.class_num):
                    npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=args.eval_threshold, use_07_metric=args.use_voc_07_metric)
                    info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
                    rec_total.update(rec, npos)
                    prec_total.update(prec, nd)
                    ap_total.update(ap, 1)

                mAP = ap_total.average
                info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
                info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                    val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average)
                print(info)
                logging.info(info)

                if mAP > best_mAP:
                    best_mAP = mAP
                    saver_best.save(sess, args.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                                    epoch, int(__global_step), best_mAP, val_loss_total.average, __lr))

                writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_recall', rec_total.average), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', prec_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average), global_step=epoch)

    ##############################################################################################################
    
if __name__ == "__main__":
    train_fn()
