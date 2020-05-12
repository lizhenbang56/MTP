import os
import math
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def vis(im_path, im_x_crop, score, cls, ctr, box):
    """
    :param im_path:
    :type im_path:
    :param im_x_crop:
    :type im_x_crop:  shape=(303, 303, 3)
    :param score: 总分
    :type score: shape=(361,)
    :param cls: 分类得分
    :type cls: shape=(361, 1)
    :param ctr:
    :type ctr: shape=(361, 1)
    :param box: xyxy
    :type box: shape=(361, 4)
    :return:
    :rtype:
    """
    def reshape_resize(array):
        resize = False
        array = array.reshape(score_map_size, score_map_size) * 255
        if resize:
            array = cv2.resize(array, (image_size, image_size))
        array = array.astype(np.uint8)
        return array

    im_x_crop = cv2.cvtColor(im_x_crop, cv2.COLOR_RGB2BGR)
    video_name, im_name = im_path.split('/')[-2:]
    image_size = im_x_crop.shape[0]
    score_map_size = int(math.sqrt(len(score)))
    score = reshape_resize(score)
    cls = reshape_resize(cls[:, 0])
    ctr = reshape_resize(ctr[:, 0])

    """
    '''可视化候选框'''
    ids = py_cpu_nms(box, score, 0.3)  # ids = score.argsort()[-100:][::-1]
    top_box = box[ids]
    for b in top_box:
        x1, y1, x2, y2 = b
        im_x_crop = cv2.rectangle(im_x_crop, (x1, y1), (x2, y2),
                                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                                  thickness=2)
    """

    plt.subplot(221)
    plt.imshow(im_x_crop)
    plt.subplot(222)
    plt.imshow(score)
    plt.subplot(223)
    plt.imshow(cls)
    plt.subplot(224)
    plt.imshow(ctr)
    save_dir = os.path.join('/tmp/vis', video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, im_name)
    plt.savefig(save_path)
    save_path_1 = os.path.join(save_dir, 'im_' + im_name)
    cv2.imwrite(save_path_1, cv2.cvtColor(im_x_crop, cv2.COLOR_RGB2BGR))
    return
