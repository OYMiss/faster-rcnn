from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from layers.anchor_generation_layer import generate_anchors
from layers.proposal_computation_layer import compute_proposals

from layers.region_proposal_network import RegionProposalNetwork

from torchvision.ops import RoIAlign, RoIPool


class RCNN(nn.Module):
    def __init__(self, feature_extractor, conv_to_head, config):
        nn.Module.__init__(self)
        # total/rpn
        self.mode = 'test'
        # something
        self.anchors = None
        self.rois_list = None  # for train
        # layer
        self.feature_extractor = feature_extractor
        self.conv_to_head = conv_to_head

        # config
        self.feature_stride = config['feature_stride']
        self.feature_compress = config['feature_compress']
        self.num_feature_channel = config['num_feature_channel']
        self.num_fc7_channel = config['num_fc7_channel']
        self.num_rpn_channel = config['num_rpn_channel']
        self.num_anchor = config['num_anchor']
        self.score_top_n = config['score_top_n']
        self.nms_top_n = config['nms_top_n']
        self.nms_thresh = config['nms_thresh']
        self.pool_out_size = config['pool_out_size']
        self.num_class = config['num_class']

        # radios = (0.5, 1, 2), scales = (8, 16, 32)
        self.radios = config['radios']
        self.scales = config['scales']

        self.rpn = RegionProposalNetwork(self.num_feature_channel,
                                         self.num_rpn_channel, self.num_anchor)
        self.roi_pool = RoIAlign(self.pool_out_size, self.feature_compress, -1)

        self.classification_head = nn.Linear(self.num_fc7_channel, self.num_class)
        self.regression_head = nn.Linear(self.num_fc7_channel, self.num_class * 4)

    def rpn_mode(self):
        self.mode = 'rpn'

    def total_mode(self):
        self.mode = 'total'

    def test_mode(self):
        self.mode = 'test'

    def forward(self, image):

        torch.backends.cudnn.benchmark = False

        # 1. net_conv
        feature = self.feature_extractor(image)

        # s1. generate_anchors
        if self.anchors is None:
            feature_width, feature_height = feature.shape[2:]
            self.anchors = generate_anchors(feature_width, feature_height, self.feature_stride,
                                            radios=self.radios, scales=self.scales)

        # 2. region_proposal_network
        # rois_list = List[Tensor(-1 ,4)]
        rpn_score, rpn_delta = self.rpn(feature, self.anchors)

        if self.mode == 'rpn':
            return rpn_score, rpn_delta

        rois_list = compute_proposals(rpn_score, rpn_delta, self.anchors,
                                      self.feature_stride,
                                      self.score_top_n, self.nms_top_n, self.nms_thresh)
        # use for train/loss
        if self.mode == 'total':
            # from layers.box_annotation_layer import annotate_proposals
            # label_list_, fg_delta_list_, index_list_ \
            #     = annotate_proposals(rois_list, gt_bbox, gt_label)
            self.rois_list = rois_list

        # 3. roi_pool
        pool = self.roi_pool(feature, rois_list)

        # benchmark because now the input size are fixed
        torch.backends.cudnn.benchmark = True

        # 4. conv_to_head
        fc7 = self.conv_to_head(pool)

        # 5. head
        score = self.classification_head(fc7)
        bbox = self.regression_head(fc7)
        return score, bbox.reshape(-1, self.num_class, 4)
