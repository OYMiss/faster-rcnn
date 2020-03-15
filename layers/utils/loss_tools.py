import torch


def smooth_l1_loss(x, y, sigma=3):
    # beta = 1 / sigma^2
    beta = 1 / (sigma * sigma)
    d = torch.abs(x - y)
    loss = torch.where(d < beta, 0.5 * d ** 2 / beta, d - 0.5 * beta)
    return loss.mean()

