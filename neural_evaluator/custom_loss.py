# pylint: disable=not-callable, no-member, no-name-in-module

import torch
from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.softmax = nn.Softmax(dim=1)
        self.eps = 1e-12

    def forward(self, policies, wins, target_policies, target_wins):
        assert policies.shape[0] == wins.shape[0]
        assert target_policies.shape[0] == target_wins.shape[0]
        assert wins.shape[0] == target_wins.shape[0]

        mse_value = self.mse_loss(wins, target_wins)
        kl_value = self.kl_loss(torch.log(policies + self.eps), target_policies)

        loss = mse_value + self.weight * kl_value
        return loss
