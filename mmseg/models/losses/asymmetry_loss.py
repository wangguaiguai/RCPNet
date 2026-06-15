import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES

@LOSSES.register_module()
class AsymmetryLoss(nn.Module):

    def __init__(self,
                 gamma_pos=1.0,
                 gamma_neg=4.0,
                 loss_weight=1.0,
                 reduction='mean'):

        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_name = 'loss_alf'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        reduction = reduction_override if reduction_override else self.reduction

        num_classes = pred.shape[1]

        pred = F.softmax(pred, dim=1)

        target = target.squeeze(1)

        valid_mask = (target != ignore_index)

        total_loss = 0.0

        for c in range(num_classes):

            p = pred[:, c]
            gt = (target == c).float()

            p = p[valid_mask]
            gt = gt[valid_mask]

            pos_loss = - gt * ((1 - p) ** self.gamma_pos) * torch.log(p + 1e-6)
            neg_loss = - (1 - gt) * (p ** self.gamma_neg) * torch.log(1 - p + 1e-6)

            total_loss += (pos_loss + neg_loss).mean()

        if reduction == 'mean':
            loss = total_loss / num_classes
        else:
            loss = total_loss

        return loss * self.loss_weight


