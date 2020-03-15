from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class RegionProposalNetwork(nn.Module):
    def __init__(self, num_feature_channel, num_rpn_channel, num_anchor):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(num_feature_channel, num_rpn_channel, 3, padding=1)
        self.cls_layer = nn.Conv2d(num_rpn_channel, 2 * num_anchor, 1)
        self.bbox_layer = nn.Conv2d(num_rpn_channel, 4 * num_anchor, 1)

    def forward(self, feature: Tensor, anchors: Tensor):
        # anchors(h, w, a, 4)
        feature = F.relu(self.conv(feature))
        n, _, h, w = feature.shape
        # score(n, 2a, h, w)
        # bbox(n, 4a, h, w)
        score = self.cls_layer(feature)
        delta = self.bbox_layer(feature)
        score = score.permute(0, 2, 3, 1).reshape(n, h, w, -1, 2)
        delta = delta.permute(0, 2, 3, 1).reshape(n, h, w, -1, 4)
        # anchors(h, w, a, 4)
        # score(n, h, w, a, 2)
        # bbox(n, h, w, a, 4)
        return score, delta
        # return self.create_proposal(score, delta, anchors)


if __name__ == '__main__':
    import time
    from layers.anchor_generation_layer import generate_anchors
    anchors_ = generate_anchors(64, 64, 16)
    feature_ = torch.rand((8, 512, 64, 64))

    t = time.time()
    rpn = RegionProposalNetwork(512, 512, 9)
    score_, delta_ = rpn.forward(feature_, anchors_)
    print(time.time() - t)
    print(score_.shape)
    print(delta_.shape)
