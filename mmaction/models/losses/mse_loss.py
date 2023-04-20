# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class MSELoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
    
    def _forward(self, x1, x2, **kwargs):
        return F.mse_loss(x1, x2)