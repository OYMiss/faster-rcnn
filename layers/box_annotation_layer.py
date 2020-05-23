from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
from layers.utils import bbox_tools
from torch import Tensor
from typing import List, Tuple


def annotate_anchors(anchors, gt_bbox,
                     feature_stride,
                     pos_thresh=0.7, neg_thresh=0.3,
                     num_sample=256,  pos_ratio=0.5
                     ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]]:
    # anchors(h, w, 9, 4)
    # gt_bbox(n, c, 4)
    # return label(n, s), fg_delta(n, s1, 4), index(n, (s1,s2))
    # feature_stride 为了计算 h, w 使得不超过图像范围

    pos_num, neg_num = int(num_sample * pos_ratio), int(num_sample * (1 - pos_ratio))
    w, h = anchors.shape[0] * feature_stride, anchors.shape[1] * feature_stride
    n, c, _ = gt_bbox.shape
    anchors = torch.reshape(anchors, (-1, 4))
    a = anchors.shape[0]

    # iou(n, c, a)
    # iou = bbox_tools.box_iou(anchors, gt_bbox)
    iou = torchvision.ops.box_iou(gt_bbox.reshape(-1, 4), anchors).reshape(n, c, a)
    max_iou, max_iou_index = iou.max(dim=1)

    # label(n, a)
    label = torch.full((n, a), -1).long()
    # max_iou(n, a) gt_index(n, a)
    label[max_iou > pos_thresh] = 1
    label[max_iou < neg_thresh] = 0
    # outside_mask(a,)
    outside_mask = torch.where(
        (anchors[:, 0] < 0) | (anchors[:, 1] < 0) |
        (anchors[:, 2] >= h) | (anchors[:, 3] >= w)
    )[0]
    # -1 don't care
    label[:, outside_mask] = -1

    # gt_proposal(n, a, 4)
    gt_proposal = gt_bbox[torch.arange(n).unsqueeze(1), max_iou_index]

    label_list = []
    fg_delta_list = []
    index_list = []
    for i in range(n):
        pos_i = torch.where(label[i] == 1)[0]
        neg_i = torch.where(label[i] == 0)[0]
        pos_i = pos_i[torch.randperm(pos_i.shape[0])[0:min(pos_i.shape[0], pos_num)]]
        neg_i = neg_i[torch.randperm(neg_i.shape[0])[0:min(neg_i.shape[0], neg_num)]]
        index = torch.cat((pos_i, neg_i))
        sampled_labels = label[i, index]
        sampled_fg_deltas = bbox_tools.to_delta(anchors[pos_i], gt_proposal[i, pos_i])
        label_list.append(sampled_labels)
        fg_delta_list.append(sampled_fg_deltas)
        index_list.append((pos_i, neg_i))

    return label_list, fg_delta_list, index_list


def annotate_proposals(proposals, gt_bbox, gt_label,
                       fg_thresh=0.5, bg_thresh=(0.1, 0.5),
                       fg_ratio=0.25
                       ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]]:
    assert fg_thresh == bg_thresh[1]
    # proposals List(p, 4)
    # gt_bbox(n, c, 4)
    # gt_label(n, c)

    n, c, _ = gt_bbox.shape
    label_list = []
    fg_delta_list = []
    index_list = []
    acc = 0
    for i in range(n):
        # iou(c, p)
        iou = torchvision.ops.box_iou(gt_bbox[i], proposals[i])
        # max_iou(p)
        max_iou, max_iou_index = iou.max(dim=0)
        # label(p)
        label = gt_label[i, max_iou_index]
        label[max_iou < bg_thresh[1]] = 0
        # don't care
        label[max_iou < bg_thresh[0]] = -1
        # gt_proposal(p, 4)
        gt_proposal = gt_bbox[i, max_iou_index]

        fg_i = torch.where(label >= 1)[0]
        bg_i = torch.where(label == 0)[0]

        index = torch.cat((fg_i, bg_i))
        sampled_labels = label[index]
        sampled_fg_deltas = bbox_tools.to_delta(proposals[i][fg_i], gt_proposal[fg_i])

        label_list.append(sampled_labels)
        fg_delta_list.append(sampled_fg_deltas)
        fg_i += acc
        bg_i += acc
        acc += len(proposals[i])
        index_list.append((fg_i, bg_i))

    return label_list, fg_delta_list, index_list


def draw(anchors_, gt_boxes_):
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    img = np.ones((1024, 1024, 3))
    point_color = (0, 255, 0)
    for i in range(9):
        x1 = anchors_[30, 30, i, 0]
        y1 = anchors_[30, 30, i, 1]
        x2 = anchors_[30, 30, i, 2]
        y2 = anchors_[30, 30, i, 3]
        cv2.rectangle(img, (x1, y1), (x2, y2), point_color)

    point_color = (255, 0, 0)
    for i in range(4):
        x1 = gt_boxes_[1, i, 0]
        y1 = gt_boxes_[1, i, 1]
        x2 = gt_boxes_[1, i, 2]
        y2 = gt_boxes_[1, i, 3]
        cv2.rectangle(img, (x1, y1), (x2, y2), point_color)
    plt.imshow(img)
    plt.show()


def test_annotate_anchors():
    import time
    from layers.anchor_generation_layer import generate_anchors
    anchors_ = generate_anchors(64, 64, 16, radios=(0.5, 1, 2), scales=(8, 16, 32))
    torch.random.manual_seed(232)
    gt_boxes_ = torch.randint(0, 64 * 16, (10, 4, 4)).sort(dim=2)[0].float()
    print('test annotate_anchors')
    t = time.time()
    label_list_, fg_delta_list_, index_list_ = \
        annotate_anchors(anchors_, gt_boxes_, feature_stride=16, pos_thresh=0.7)
    print(time.time() - t)
    # draw(anchors_, gt_boxes_)
    print(label_list_[0].shape)
    print(fg_delta_list_[0].shape)
    print(index_list_[0][0].shape, index_list_[0][1].shape)


def test_annotate_proposals():
    import time
    from layers.anchor_generation_layer import generate_anchors
    from layers.region_proposal_network import RegionProposalNetwork
    from layers.proposal_computation_layer import compute_proposals

    torch.random.manual_seed(232)
    num_class = 3
    n = 10
    gt_boxes_ = torch.randint(0, 64 * 16, (n, 10, 4)).sort(dim=2)[0].float()
    gt_label_ = torch.randint(0, num_class - 1, (n, 10)).float()

    anchors_ = generate_anchors(64, 64, 16)
    feature_ = torch.rand((n, 512, 64, 64))
    rpn = RegionProposalNetwork(512, 512, 9)
    score_, delta_ = rpn.forward(feature_, anchors_)
    proposal_list_ = compute_proposals(score_, delta_, anchors_, 16, 100, 50, 0.7)

    print('test annotate_proposals')
    t = time.time()
    label_list_, fg_delta_list_, index_list_ = \
        annotate_proposals(proposal_list_, gt_boxes_, gt_label_)
    print(time.time() - t)
    print(label_list_[0].shape)
    print(fg_delta_list_[0].shape)
    print(index_list_[0][0].shape, index_list_[0][1].shape)


if __name__ == '__main__':
    test_annotate_anchors()
    test_annotate_proposals()
