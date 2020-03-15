from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from layers.utils import loss_tools

from layers.box_annotation_layer import annotate_anchors


def rpn_loss_layer(rpn_score, rpn_delta, label_list, fg_delta_list, index_list):
    # rpn_score (n, h, w, a, c)
    # rpn_delta (n, h, w, a, 4)
    # label_list list(n, -1)
    # fg_delta_list list(n, s1, 4)
    # index_list list((n, s1), (n, s2)))
    # return loss()
    n = rpn_score.shape[0]
    rpn_score = rpn_score.reshape(n, -1, 2)
    rpn_delta = rpn_delta.reshape(n, -1, 4)
    reg_loss, cls_loss = torch.tensor(0.), torch.tensor(0.)
    for i, (pos_i, neg_i) in enumerate(index_list):
        index = torch.cat((pos_i, neg_i))

        label = label_list[i]
        score_i = rpn_score[i, index]
        if label.shape[0] > 0:
            cls_loss += F.cross_entropy(score_i, label)

        gt_fg_delta = fg_delta_list[i]
        fg_delta = rpn_delta[i, pos_i]
        if fg_delta.shape[0] > 0:
            reg_loss += loss_tools.smooth_l1_loss(fg_delta, gt_fg_delta)
    return reg_loss + cls_loss


def compute(model, rpn_score, rpn_bbox, gt_label, gt_bbox):
    label_list, fg_delta_list, index_list = \
        annotate_anchors(model.anchors, gt_bbox, feature_stride=model.feature_stride)
    rpn_loss = rpn_loss_layer(rpn_score, rpn_bbox, label_list, fg_delta_list, index_list)
    return rpn_loss


if __name__ == '__main__':
    import time
    from layers.anchor_generation_layer import generate_anchors
    from layers.region_proposal_network import RegionProposalNetwork
    # from layers.box_annotation_layer import annotate_anchors

    _n = 10
    anchors_ = generate_anchors(64, 64, 16)
    feature_ = torch.rand((_n, 512, 64, 64), requires_grad=True)
    rpn = RegionProposalNetwork(512, 512, 9)
    rpn_score_, rpn_delta_ = rpn.forward(feature_, anchors_)
    gt_boxes_ = torch.randint(0, 64 * 16, (_n, 4, 4)).sort(dim=2)[0].float()
    label_list_, fg_delta_list_, index_list_ = annotate_anchors(anchors_, gt_boxes_, feature_stride=16)
    print(rpn_score_.shape)
    print(rpn_delta_.shape)
    t = time.time()
    loss_ = rpn_loss_layer(rpn_score_.reshape(_n, -1, 2), rpn_delta_.reshape(_n, -1, 4),
                           label_list_, fg_delta_list_, index_list_)
    # loss_.backward()
    print(time.time() - t)
    print(loss_)
