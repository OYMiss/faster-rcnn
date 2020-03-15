from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import Tensor


def to_delta(boxes: Tensor, gt_proposals: Tensor) -> Tensor:
    # proposal(a, 4)
    # boxes(a, 4)
    # return deltas(a, 4)
    # boxes(a, 4)
    # w(a, 1)
    w, h, x, y = w_h_x_y(boxes)
    # pred_w(a, 1)
    pred_w, pred_h, pred_x, pred_y = w_h_x_y(gt_proposals)

    dx, dy = pred_x - x / w, pred_y - y / h
    dw, dh = torch.log(pred_w / w), torch.log(pred_h / h)

    return torch.cat((dx, dy, dw, dh), dim=1)


def to_proposal(boxes: Tensor, deltas: Tensor) -> Tensor:
    # boxes(h, w, a, 4)
    # deltas(n, h, w, a, 4)
    # return proposal(n, -1, 4)
    n = deltas.shape[0]
    boxes = boxes.reshape(1, -1, 4)
    deltas = deltas.reshape(n, -1, 4)
    # x(1, -1, 1)
    w, h, x, y = w_h_x_y(boxes)
    # dx(n, -1, 1)
    dw, dh, dx, dy = deltas.split(1, dim=2)

    pred_x, pred_y = dx * w + x, dy * h + y
    pred_w, pred_h = torch.exp(dw) * w, torch.exp(dh) * h

    pred_boxes = make_anchor(pred_w, pred_h, pred_x, pred_y)
    return pred_boxes.reshape(n, -1, 4)


def clip_proposal(boxes, width, height):
    # boxes(n, -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, width - 1),
        boxes[:, :, 1].clamp(0, height - 1),
        boxes[:, :, 2].clamp(0, width - 1),
        boxes[:, :, 3].clamp(0, height - 1)], 2)
    return boxes


def w_h_x_y(anchor):
    # anchor must be (x, ..., 4)
    # return w(x, ..., 1)
    dim = len(anchor.shape) - 1
    x1, y1, x2, y2 = anchor.split(1, dim=dim)
    # anchor.select(dim, 0)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    x = x1 + (w - 1) / 2
    y = y1 + (h - 1) / 2
    return w, h, x, y


def make_anchor(w, h, x, y):
    # w must be (x, ..., 1)
    # return anchor(x, ..., 4)
    dim = len(w.shape) - 1
    anchor = torch.cat([x - 0.5 * (w - 1),
                        y - 0.5 * (h - 1),
                        x + 0.5 * (w - 1),
                        y + 0.5 * (h - 1)], dim)
    return anchor


'''
def box_area(boxes: Tensor):
    # boxes (x, ..., 4)
    d = len(boxes.shape) - 1
    return (boxes.select(d, 2) - boxes.select(d, 0)) * (boxes.select(d, 3) - boxes.select(d, 1))


def box_iou(anchors, gt_bbox):
    # anchors(a, 4)
    # gt_bbox(n, c, 4)
    # return iou(n, c, a)
    n, c, _ = gt_bbox.shape
    # anchors(1, a, 4)
    anchors = anchors.reshape((1, -1, 4))
    # gt_bbox(nc, 1, 4)
    gt_bbox = gt_bbox.reshape((-1, 1, 4))

    # boxes1 = anchors.reshape(-1, 4)
    # boxes2 = gt_bbox.reshape(-1, 4)
    # area1 = box_area(boxes1)
    # area2 = box_area(boxes2)
    #
    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    #
    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    #
    # iou2 = inter / (area1[:, None] + area2 - inter)

    xy1 = torch.max(anchors.reshape(-1, 4)[:, :2], gt_bbox.reshape(-1, 4)[:, None, :2])  # [N,M,2]
    xy2 = torch.min(anchors.reshape(-1, 4)[:, 2:], gt_bbox.reshape(-1, 4)[:, None, 2:])  # [N,M,2]

    # xy1 = torch.max(anchors[:, :, :2], gt_bbox[:, :, :2])
    # xy2 = torch.min(anchors[:, :, 2:], gt_bbox[:, :, 2:])
    area_i = (xy2 - xy1)[:, :, 0] * (xy2 - xy1)[:, :, 1] * torch.all(xy1 < xy2, dim=2).float()

    # error = torch.sum(inter - area_i)

    area_a = box_area(anchors)
    area_b = box_area(gt_bbox)

    # iou(nc, a)
    iou = area_i / (area_a + area_b - area_i)
    return iou.reshape(n, c, -1)
'''

if __name__ == '__main__':
    from layers.anchor_generation_layer import generate_anchors
    import time
    t = time.time()
    a_ = generate_anchors(64, 64, 16)
    b_ = torch.softmax(torch.stack((a_, a_)), dim=0)
    c_ = to_proposal(a_, b_)
    d_ = clip_proposal(c_, 64, 64)
    print(time.time() - t)
