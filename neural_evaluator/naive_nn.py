import torch
import torch.nn.functional as F
import torch.nn as nn


class NaiveNet(torch.nn.Module):
    def __init__(self, num_rows: int, num_cols: int):
        super(NaiveNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 20, (3, 3))
        self.conv2 = nn.Conv2d(20, 40, (3, 3))

        self.flat_size = (num_rows - 4) * (num_cols - 4) * 40
        self.linear = torch.nn.Linear(in_features=self.flat_size, out_features=num_cols)
        self.linear2 = torch.nn.Linear(in_features=self.flat_size, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.linear.in_features)
        column_x = self.linear(x)
        score_x = self.linear2(x)
        return column_x, score_x
