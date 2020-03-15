from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from rcnn import RCNN
from resnet import resnet18

from layers import rcnn_loss_layer
from layers import rpn_loss_layer

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
    'num_class': 5,  # 这里的标签数量包含背景（0），前景类别从 1 开始。
    'radios': (0.5, 1, 2),
    'scales': (4, 8, 16),
    # 'scales': (8, 16, 32),
}

n = 2
model = RCNN(feature_extractor, conv_to_head, config)
image = torch.rand((n, 3, 128, 128))
gt_bbox = torch.randint(0, 8 * 16, (n, 8, 4)).sort(dim=2)[0].float()
gt_label = torch.randint(1, config['num_class'], (n, 8))

model.rpn_mode()
rpn_score, rpn_bbox = model.forward(image)
print(rpn_score.shape, rpn_bbox.shape)
rpn_loss = rpn_loss_layer.compute(model, rpn_score, rpn_bbox, gt_label, gt_bbox)
rpn_loss.backward()

model.total_mode()
score, bbox = model.forward(image)
print(score.shape, bbox.shape)
rcnn_loss = rcnn_loss_layer.compute(model, score, bbox, gt_label, gt_bbox)
rcnn_loss.backward()

