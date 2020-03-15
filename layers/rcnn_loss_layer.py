from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from layers.utils import loss_tools
from layers.box_annotation_layer import annotate_proposals


def rcnn_loss_layer(score, delta, label_list, fg_delta_list, index_list):
    # score (p, c)
    # delta (p, c, 4)
    # label_list list(n, l) sum(l) == k
    # fg_delta_list list(n, l1, 4)
    # index_list List[Tuple[Tensor, Tensor]]
    # return loss()
    p, num_class = score.shape[:2]
    reg_loss, cls_loss = torch.tensor(0.), torch.tensor(0.)
    n = len(index_list)

    # labels = torch.cat(label_list)
    for i in range(n):
        label = label_list[i]
        fg_i, bg_i = index_list[i]
        index = torch.cat((fg_i, bg_i))
        # (k,)
        cls_loss += F.cross_entropy(score[index], label)

        # (l1, 4)
        fg_delta_i = fg_delta_list[i]
        for j in range(1, num_class + 1):
            index_j = torch.where(label == j)
            x = fg_i[index_j]
            if x.shape[0] == 0:
                continue
            pred_delta_j = delta[x, j]
            gt_delta_j = fg_delta_i[index_j]
            if pred_delta_j.shape[0] > 0:
                reg_loss += loss_tools.smooth_l1_loss(pred_delta_j, gt_delta_j)

    return reg_loss + cls_loss


def compute(model, score, bbox, gt_label, gt_bbox):
    label_list, fg_delta_list, index_list = \
        annotate_proposals(model.rois_list, gt_bbox, gt_label)
    rcnn_loss = rcnn_loss_layer(score, bbox, label_list, fg_delta_list, index_list)
    return rcnn_loss


if __name__ == '__main__':
    import time
    from rcnn import RCNN
    from resnet import resnet18
    from torch import nn

    resnet = resnet18()

    feature_extractor = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3
    )

    # noinspection PyUnresolvedReferences
    conv_to_head = nn.Sequential(
        resnet.layer4,
        resnet.avgpool,
        nn.Flatten(),
    )

    config = {
        'feature_stride': 16,
        'feature_compress': 1 / 16,
        'num_feature_channel': 256,
        'num_fc7_channel': 512,
        'num_rpn_channel': 512,
        'num_anchor': 9,
        'score_top_n': 100,
        'nms_top_n': 50,
        'nms_thresh': 0.7,
        'pool_out_size': 8,
        'num_class': 5,
        'radios': (0.5, 1, 2),
        'scales': (8, 16, 32),
    }

    model_ = RCNN(feature_extractor, conv_to_head, config)

    n_ = 10
    image = torch.rand((n_, 3, 128, 128))

    model_.total_mode()
    # score_(k, num_cls) delta_(k, num_cls, 4)
    score_, delta_ = model_.forward(image)
    print(score_.shape, delta_.shape)

    gt_bbox_ = torch.randint(0, 8 * 16, (n_, 8, 4)).sort(dim=2)[0].float()
    gt_label_ = torch.randint(1, config['num_class'], (n_, 8))
    label_list_, fg_delta_list_, index_list_ = annotate_proposals(model_.rois_list, gt_bbox_, gt_label_)
    print(label_list_[0].shape, fg_delta_list_[0].shape, index_list_[0][0].shape, index_list_[0][1].shape)
    t = time.time()
    loss_ = rcnn_loss_layer(score_, delta_, label_list_, fg_delta_list_, index_list_)
    print(time.time() - t)
    print(loss_)
