import torch
from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self, weight=1):
        super(AlphaLoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.softmax = nn.Softmax(dim=1)
        self.eps = 1e-12

    def forward(self, policies, wins, target_policies, target_wins):
        assert policies.shape[0] == wins.shape[0]
        assert target_policies.shape[0] == target_wins.shape[0]
        assert wins.shape[0] == target_wins.shape[0]

        reg_wins = self.softmax(wins)[:, 1]
        reg_policies = self.softmax(policies)
        mse = self.mse_loss(reg_wins, target_wins)
        kl = self.kl_loss(torch.log(reg_policies + self.eps), target_policies)

        loss = mse + self.weight * kl
        return loss
