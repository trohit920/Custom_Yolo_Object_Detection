# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import random

from new_args import parse_anchors, read_class_names
from new_train import gpu_nms
from new_train import letterbox_resize

from new_train import yolov3

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)



parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/pitta.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint_sent/yolov3.ckpt",
                    help="The path of the weights to restore.")
# parser.add_argument("--restore_path", type=str, default="./checkpoint/model-epoch_10_step_4014_loss_1.2792_lr_0.0001",
#                     help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

img_ori = cv2.imread(args.input_image)
if args.letterbox_resize:
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
else:
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)
