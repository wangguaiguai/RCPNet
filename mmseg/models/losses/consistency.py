import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class ConsistencyLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()

        self.loss_weight = loss_weight
        self.loss_name = 'loss_cons'

    def forward(self, z_swin, z_mpd, R=None):

        z_swin = F.normalize(z_swin, dim=1)
        z_mpd = F.normalize(z_mpd, dim=1)

        diff = (z_swin - z_mpd) ** 2
        diff = diff.mean(dim=1, keepdim=True)
        loss = diff.mean()

        return loss * self.loss_weight

