from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from layers.utils import bbox_tools


def generate_anchors(feature_width, feature_height, feature_stride, radios=(0.5, 1, 2), scales=(8, 16, 32)):
    # return anchors(h, w, 9, 4)
    base_anchors = generate_anchors_at_0_0(16, radios, scales)

    x = torch.arange(0, feature_width) * feature_stride
    y = torch.arange(0, feature_height) * feature_stride
    # shift_(feature_width, feature_height)
    shift_x, shift_y = torch.meshgrid(x, y)
    # shift_cord(feature * width, 4)
    shift_cord = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)]).T
    anchors = base_anchors.reshape((1, -1, 4)) + shift_cord.reshape((-1, 1, 4))
    anchors = torch.reshape(anchors, (feature_height, feature_width, -1, 4))
    # anchors(h, w, 9, 4)
    return anchors


def generate_anchors_at_0_0(base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    base_anchor = torch.tensor([1., 1., base_size, base_size]) - 1
    ratios, scales = torch.tensor(ratios), torch.tensor(scales)
    w_, h_, x, y = bbox_tools.w_h_x_y(base_anchor)
    w = torch.round(torch.sqrt(w_ * h_ / ratios))
    h = torch.round(ratios * w)
    w = (w.reshape(-1, 1) * scales).reshape(-1, 1)
    h = (h.reshape(-1, 1) * scales).reshape(-1, 1)
    anchors = bbox_tools.make_anchor(w, h, x, y)
    return anchors


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors_at_0_0()
    b = generate_anchors(64, 64, 16)
    print(time.time() - t)
    # print(a)
    print(b)
