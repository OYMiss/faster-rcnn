from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torch.nn.functional as F

from layers.utils import bbox_tools

from typing import List
from torch import Tensor


def compute_proposals(score, delta, anchors, feature_stride,
                      score_top_n=100, nms_top_n=50, nms_thresh=0.7
                      ) -> List[Tensor]:
    # anchors(h, w, a, 4)
    # score(n, h, w, a, 2)
    # delta(n, h, w, a, 4)
    # return proposal_list List[Tensor(-1, 4)]
    n, h, w = score.shape[0:3]
    proposal = bbox_tools.to_proposal(anchors, delta)
    # proposal(n, -1, 4)
    proposal = bbox_tools.clip_proposal(proposal, h * feature_stride, w * feature_stride)
    # fg_prod(n, -1)
    fg_prod = F.softmax(score, dim=4)[:, :, :, :, 1].reshape(n, -1)

    _, mask = fg_prod.sort(descending=True)
    mask = mask[:, 0:score_top_n]
    proposal = proposal[torch.arange(n).reshape(-1, 1), mask]
    fg_prod = fg_prod[torch.arange(n).reshape(-1, 1), mask]

    proposal_list = []
    for i in range(n):
        mask = torchvision.ops.nms(proposal[i], fg_prod[i], nms_thresh)
        mask = mask[0:nms_top_n]
        proposal_list.append(proposal[i, mask])

    return proposal_list


if __name__ == '__main__':
    import time
    from layers.anchor_generation_layer import generate_anchors
    from layers.region_proposal_network import RegionProposalNetwork

    anchors_ = generate_anchors(64, 64, 16)
    feature_ = torch.rand((8, 512, 64, 64))
    rpn = RegionProposalNetwork(512, 512, 9)
    score_, delta_ = rpn.forward(feature_, anchors_)
    score_top_n_ = 100
    nms_top_n_ = 20
    nms_thresh_ = 0.7

    t = time.time()
    proposal_list_ = compute_proposals(score_, delta_, anchors_, 16, score_top_n_, nms_top_n_, nms_thresh_)
    print(time.time() - t)
    print(proposal_list_[0].shape)
