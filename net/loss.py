import theano.tensor as T
import keras.backend as K
import numpy as np
import theano
from tensorflow.contrib import slim as slim


############################################################
################### SOURCE https://groups.google.com/forum/#!topic/darknet/BJFv5obe3bQ
############################################################


def overlap(x1, w1, x2, w2, side=7):
    """
    Args:
      x1: x for the first box
      w2: width for the first box
      x2 : x for the second box
      w2: width for the second box
    """
    # Find out how much percent the center is over the whole width/height
    x1 = x1 / side
    x2 = x2 / side
    l1 = x1 - w1 * w1 / 2
    l2 = x2 - w2 * w2 / 2
    left = T.switch(T.lt(l1, l2), l2, l1)
    r1 = x1 + w1 * w1 / 2
    r2 = x2 + w2 * w2 / 2
    right = T.switch(T.lt(r1, r2), r1, r2)
    return right - left


def box_intersection(a, b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the intersection area of each two boxes
    """
    w = overlap(a[:, :, 0], a[:, :, 3], b[:, :, 0], b[:, :, 3])
    h = overlap(a[:, :, 1], a[:, :, 2], b[:, :, 1], b[:, :, 2])
    w = T.switch(T.lt(w, 0), 0, w)
    h = T.switch(T.lt(h, 0), 0, h)
    area = w * h
    return area


def box_union(a, b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the union area of each two boxes
    """
    i = box_intersection(a, b)
    area_a = a[:, :, 2] * a[:, :, 3]
    area_b = b[:, :, 2] * b[:, :, 3]
    u = area_a * area_a + area_b * area_b - i
    return u


def box_iou(a, b):
    """
    Args:
      a: the first box, a n*4 tensor
      b: the second box,another n*4 tensor
    Returns:
      area: n*1 tensor, indicating the intersection over union of each two boxes
    Notes: boxes with 0 union should has 0 iou, don't know why orignal yolo ignores this situation
    """
    # the net and groud truth are all the square root of height and width
    u = box_union(a, b)
    i = box_intersection(a, b)
    iou = T.switch(T.eq(u, 0), 0, i / u)
    return iou


def box_mse(a, b):
    mse = T.sum(T.square(a - b), axis=2)
    return mse


def custom_loss_2(y_true, y_pred):
    """
    Args:
      y_true: ground truth tensor
      y_pred: tensor predicted by the network
    Returns:
      loss: the sumed mse loss, details of this loss function can be found in yolo paper
    """
    # the ground truth x is the offset within a cell, but we need actual ground truth x to calculate the overlap
    offset = []
    for i in range(7):
        for j in range(7):
            offset.append(j)
            offset.append(i)
            offset.extend([0] * 3)
            offset.append(j)
            offset.append(i)
            offset.extend([0] * 23)
    y_pred_offset = y_pred + np.asarray(offset)
    y_true_offset = y_true + np.asarray(offset)

    loss = 0.0
    y_pred = y_pred.reshape((y_pred.shape[0], 49, 30))
    y_true = y_true.reshape((y_true.shape[0], 49, 30))
    y_pred_offset = y_pred_offset.reshape((y_pred_offset.shape[0], 49, 30))
    y_true_offset = y_true_offset.reshape((y_true_offset.shape[0], 49, 30))

    a_offset = y_pred_offset[:, :, 0:4]
    b_offset = y_pred_offset[:, :, 5:9]
    gt_offset = y_true_offset[:, :, 0:4]

    a = y_pred[:, :, 0:4]
    b = y_pred[:, :, 5:9]
    gt = y_true[:, :, 0:4]

    # iou bewteen box a and gt
    iou_a_gt = box_iou(a_offset, gt_offset)
    # don't want iou has influence on x,y,h,w, x,y,h,w are only infected by gt value
    iou_a_gt = theano.gradient.disconnected_grad(iou_a_gt)

    # iou bewteen box b and gt
    iou_b_gt = box_iou(b_offset, gt_offset)
    iou_b_gt = theano.gradient.disconnected_grad(iou_b_gt)

    # mse bewteen box a and gt
    mse_a_gt = box_mse(a_offset, gt_offset)

    # mse bewteen box b and gt
    mse_b_gt = box_mse(b_offset, gt_offset)

    # mask is either 0 or 1, 1 indicates box b has a higher iou with gt than box a
    mask = T.switch(T.lt(iou_a_gt, iou_b_gt), 1, 0)

    # if two boxes both have 0 iou with ground truth, we blame the one with higher mse with gt
    # It feels like hell to code like this,f**k!
    mask_iou_zero = T.switch(T.and_(T.le(iou_a_gt, 0), T.le(iou_b_gt, 0)), 1, 0)
    mask_mse = T.switch(T.lt(mse_a_gt, mse_b_gt), 1, 0)
    mask_change = mask_iou_zero * mask_mse
    mask = mask + mask_change
    mask = theano.gradient.disconnected_grad(mask)

    # loss bewteen box a and gt
    loss_a_gt = T.sum(T.square(a - gt), axis=2) * 5

    # loss bewteen box b and gt
    loss_b_gt = T.sum(T.square(b - gt), axis=2) * 5

    # use mask to add the loss from the box with higher iou with gt
    loss = loss + y_true[:, :, 4] * (1 - mask) * loss_a_gt
    loss = loss + y_true[:, :, 4] * mask * loss_b_gt

    # confident loss bewteen a and gt
    closs_a_gt = T.square(iou_a_gt * y_true[:, :, 4] - y_pred[:, :, 4])
    # confident loss bewteen b and gt
    closs_b_gt = T.square(iou_b_gt * y_true[:, :, 4] - y_pred[:, :, 9])

    loss = loss + closs_a_gt * (1 - mask) * y_true[:, :, 4]
    loss = loss + closs_b_gt * mask * y_true[:, :, 4]

    # if the cell has no obj, confidence loss should be halved
    loss = loss + closs_a_gt * (1 - y_true[:, :, 4]) * 0.5
    loss = loss + closs_b_gt * (1 - y_true[:, :, 4]) * 0.5

    # add loss for the conditioned classification error
    loss = loss + T.sum(T.square(y_pred[:, :, 10:30] - y_true[:, :, 10:30]), axis=2) * y_true[:, :, 4]

    # sum for each cell
    loss = T.sum(loss)

    # mean for each image
    # loss = T.mean(loss)

    return loss

# def overlap(x1, w1, x2, w2, side=7):
#     """
#     Args:
#       x1: x for the first box
#       w2: width for the first box
#       x2 : x for the second box
#       w2: width for the second box
#     """
#     # Find out how much percent the center is over the whole width/height
#     x1 = x1 / side
#     x2 = x2 / side
#     l1 = x1 - w1 * w1 / 2
#     l2 = x2 - w2 * w2 / 2
#     left = K.switch(K.less(l1, l2), l2, l1)
#     # left = K.switch(tf.less(l1, l2), l2, l1)
#     r1 = x1 + w1 * w1 / 2
#     r2 = x2 + w2 * w2 / 2
#     right = K.switch(K.less(r1, r2), r1, r2)
#     return right - left
#
#
# # def overlap(x1, w1, x2, w2):
# #     l1 = x1 - w1 / 2.
# #     l2 = x2 - w2 / 2.
# #     left = max(l1, l2)
# #     r1 = x1 + w1 / 2.
# #     r2 = x2 + w2 / 2.
# #     right = min(r1, r2)
# #     return right - left
#
#
# def box_intersection(a, b):
#     """
#     Args:
#       a: the first box, a n*4 tensor
#       b: the second box,another n*4 tensor
#     Returns:
#       area: n*1 tensor, indicating the intersection area of each two boxes
#     """
#     w = overlap(a[:, :, 0], a[:, :, 3], b[:, :, 0], b[:, :, 3])
#     h = overlap(a[:, :, 1], a[:, :, 2], b[:, :, 1], b[:, :, 2])
#     w = K.switch(K.less(w, K.zeros((None, 49))), K.zeros((None, 49)), w)
#     h = K.switch(K.less(h, 0), 0, h)
#     area = w * h
#     return area
#
#
# def box_union(a, b):
#     """
#     Args:
#       a: the first box, a n*4 tensor
#       b: the second box,another n*4 tensor
#     Returns:
#       area: n*1 tensor, indicating the union area of each two boxes
#     """
#     i = box_intersection(a, b)
#     area_a = a[:, :, 2] * a[:, :, 3]
#     area_b = b[:, :, 2] * b[:, :, 3]
#     u = area_a * area_a + area_b * area_b - i
#     return u
#
#
# def box_iou(a, b):
#     """
#     Args:
#       a: the first box, a n*4 tensor
#       b: the second box,another n*4 tensor
#     Returns:
#       area: n*1 tensor, indicating the intersection over union of each two boxes
#     Notes: boxes with 0 union should has 0 iou, don't know why orignal yolo ignores this situation
#     """
#     # the net and groud truth are all the square root of height and width
#     u = box_union(a, b)
#     i = box_intersection(a, b)
#     iou = K.switch(K.equal(u, 0), 0, i / u)
#     return iou
#
#
# def box_mse(a, b):
#     mse = K.sum(K.square(a - b), axis=2)
#     return mse
#
#
# class CustomLoss:
#     def __init__(self, side=7, boxes=2, classes=20):
#         self.__side = side
#         self.__boxes = boxes
#         self.__classes = classes
#
#     def __call__(self, y_true, y_pred):
#
#         """
#         Args:
#           y_true: ground truth tensor
#           y_pred: tensor predicted by the network
#         Returns:
#           loss: the sumed mse loss, details of this loss function can be found in yolo paper
#         """
#         # the ground truth x is the offset within a cell, but we need actual ground truth x to calculate the overlap
#         offset = []
#         for i in range(7):
#             for j in range(7):
#                 offset.append(j)
#                 offset.append(i)
#                 offset.extend([0] * 3)
#                 offset.append(j)
#                 offset.append(i)
#                 offset.extend([0] * 23)
#         y_pred_offset = y_pred + np.asarray(offset)
#         y_true_offset = y_true + np.asarray(offset)
#
#         loss = 0.0
#         # y_pred = y_pred.reshape((y_pred.shape[0], 49, 30))
#         # y_true = y_true.reshape((y_true.shape[0], 49, 30))
#         y_pred = K.reshape(y_pred, (-1, 49, 30))
#         y_true = K.reshape(y_true, (-1, 49, 30))
#
#         # y_pred_offset = y_pred_offset.reshape((y_pred_offset.shape[0], 49, 30))
#         # y_true_offset = y_true_offset.reshape((y_true_offset.shape[0], 49, 30))
#         y_pred_offset = K.reshape(y_pred_offset, (-1, 49, 30))
#         y_true_offset = K.reshape(y_true_offset, (-1, 49, 30))
#
#         a_offset = y_pred_offset[:, :, 0:4]
#         b_offset = y_pred_offset[:, :, 5:9]
#         gt_offset = y_true_offset[:, :, 0:4]
#
#         a = y_pred[:, :, 0:4]
#         b = y_pred[:, :, 5:9]
#         gt = y_true[:, :, 0:4]
#
#         # iou bewteen box a and gt
#         iou_a_gt = box_iou(a_offset, gt_offset)
#         # don't want iou has influence on x,y,h,w, x,y,h,w are only infected by gt value
#         # iou_a_gt = theano.gradient.disconnected_grad(iou_a_gt) # TODO changed
#         iou_a_gt = K.stop_gradient(iou_a_gt)
#
#         # iou bewteen box b and gt
#         iou_b_gt = box_iou(b_offset, gt_offset)
#         # iou_b_gt = theano.gradient.disconnected_grad(iou_b_gt) # TODO changed
#         iou_b_gt = K.stop_gradient(iou_b_gt)
#
#         # mse bewteen box a and gt
#         mse_a_gt = box_mse(a_offset, gt_offset)
#
#         # mse bewteen box b and gt
#         mse_b_gt = box_mse(b_offset, gt_offset)
#
#         # mask is either 0 or 1, 1 indicates box b has a higher iou with gt than box a
#         mask = K.switch(K.less(iou_a_gt, iou_b_gt), 1, 0)
#
#         # if two boxes both have 0 iou with ground truth, we blame the one with higher mse with gt
#         # It feels like hell to code like this,f**k!
#         mask_iou_zero = K.switch(T.and_.logical_and(K.less_equal(iou_a_gt, 0), K.less_equal(iou_b_gt, 0)), 1,
#                                  0)  # TODO only line with theanp
#         mask_mse = K.switch(K.less(mse_a_gt, mse_b_gt), 1, 0)
#         mask_change = mask_iou_zero * mask_mse
#         mask = mask + mask_change
#         # mask = theano.gradient.disconnected_grad(mask) # TODO changed
#         mask = K.stop_gradient(mask)
#
#         # loss bewteen box a and gt
#         loss_a_gt = K.sum(K.square(a - gt), axis=2) * 5
#
#         # loss bewteen box b and gt
#         loss_b_gt = K.sum(K.square(b - gt), axis=2) * 5
#
#         # use mask to add the loss from the box with higher iou with gt
#         loss = loss + y_true[:, :, 4] * (1 - mask) * loss_a_gt
#         loss = loss + y_true[:, :, 4] * mask * loss_b_gt
#
#         # confident loss bewteen a and gt
#         closs_a_gt = K.square(iou_a_gt * y_true[:, :, 4] - y_pred[:, :, 4])
#         # confident loss bewteen b and gt
#         closs_b_gt = K.square(iou_b_gt * y_true[:, :, 4] - y_pred[:, :, 9])
#
#         loss = loss + closs_a_gt * (1 - mask) * y_true[:, :, 4]
#         loss = loss + closs_b_gt * mask * y_true[:, :, 4]
#
#         # if the cell has no obj, confidence loss should be halved
#         loss = loss + closs_a_gt * (1 - y_true[:, :, 4]) * 0.5
#         loss = loss + closs_b_gt * (1 - y_true[:, :, 4]) * 0.5
#
#         # add loss for the conditioned classification error
#         loss = loss + K.sum(K.square(y_pred[:, :, 10:30] - y_true[:, :, 10:30]), axis=2) * y_true[:, :, 4]
#
#         # sum for each cell
#         loss = K.sum(loss)
#
#         # mean for each image
#         # loss = T.mean(loss)
#
#         return loss
