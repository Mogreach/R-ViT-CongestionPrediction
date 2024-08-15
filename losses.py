import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import losses


def build_loss(opt):
    return losses.__dict__[opt['loss_type']]()

__all__ = ['L1Loss', 'MSELoss','HuberLoss','MixLoss']


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss

def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper

@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    # s=20
    # b=0.4
    # weight = 1 / (1 + torch.exp(s * abs(pred-target)))
    # s=-1*16.27979308
    # b=0.43425323
    # s=23.62928869
    # b=0.19946715
    # weight = 10 / (1 + torch.exp(-s * (abs(pred-target)-b)))
    loss = F.mse_loss(pred, target, reduction='none')
    return loss

class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



class MSELoss(nn.Module):
    def __init__(self, loss_weight=100, reduction='mean',sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


# 定义Huber Loss函数
class HuberLoss(nn.Module):
    def __init__(self, initial_delta=0.15):
        super(HuberLoss, self).__init__()
        # self.delta = delta
        self.delta = nn.Parameter(torch.tensor(initial_delta))

    def forward(self, y_pred, y_true):
        # k=-55.28699427
        # b=0.17492549
        # error = torch.abs(y_true - y_pred)
        # weighted = 100 / (1 + torch.exp(-k * (error - b)))
        # quadratic_term = 0.5 * error**2
        # linear_term = self.delta * (error - 0.5 * self.delta)
        # loss = 100*torch.where(error <= self.delta, quadratic_term, linear_term)
        # return torch.mean(loss)
        residual = y_pred - y_true
        delta = torch.abs(self.delta)
        huber_loss = torch.where(torch.abs(residual) <= delta,
                                 0.5 * residual ** 2,
                                 delta * (torch.abs(residual) - 0.5 * delta))
        return huber_loss.mean()

#quantile中位数
class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        residual = y_true - y_pred
        quantile_loss = torch.max((self.quantile - 1) * residual, self.quantile * residual)
        return torch.mean(quantile_loss)
# 自定义Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


class GHMLoss(nn.Module):
    def __init__(self, bins=10, alpha=0.75, beta=0.01):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.beta = beta

    def forward(self, predict, target):
        # 计算梯度
        gradient = torch.abs(predict - target)
        gradient_sorted, _ = torch.sort(gradient, descending=True)

        # 将梯度分成bins个区间
        n = gradient.size(0)
        edges = [i * n // self.bins for i in range(1, self.bins)]
        edges = [0] + edges + [n]
        edges = torch.tensor(edges, dtype=torch.long, device=gradient.device)
        g = [gradient_sorted[edges[i]:edges[i + 1]].mean() for i in range(self.bins)]

        # 计算权重
        weights = torch.zeros_like(gradient)
        for i in range(self.bins):
            weights[(gradient >= g[i]) & (gradient <= g[i + 1])] = 1.0 / self.bins

        loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        loss = loss * weights
        loss = loss.sum() / weights.sum()

        return loss
class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
    def forward(self, y_pred, y_true):
        qloss = QuantileLoss(0.15)
        hloss = HuberLoss()
        ghmloss = GHMLoss()
        mseloss =  MSELoss()
        return hloss(y_pred,y_true)
