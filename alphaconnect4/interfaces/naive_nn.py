# pylint: disable=too-many-instance-attributes

import torch
import torch.nn as nn


class NaiveNet(torch.nn.Module):
    def __init__(self, num_rows: int, num_cols: int):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 50, (3, 3))
        self.conv2 = nn.Conv2d(50, 100, (3, 3))
        self.dropout_1 = nn.Dropout(p=0.3, inplace=False)
        self.dropout_2 = nn.Dropout(p=0.3, inplace=False)

        self.flat_size = (num_rows - 4) * (num_cols - 4) * 100
        self.linear_p1 = torch.nn.Linear(in_features=self.flat_size, out_features=100)
        self.linear_p2 = torch.nn.Linear(in_features=100, out_features=num_cols)

        self.linear_s1 = torch.nn.Linear(in_features=self.flat_size, out_features=100)
        self.linear_s2 = torch.nn.Linear(in_features=100, out_features=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.dropout_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.dropout_2(x)

        x = x.view(-1, self.flat_size)

        policy_x = self.softmax(self.linear_p2(torch.nn.functional.relu(self.linear_p1(x))))
        score_x = self.softmax(self.linear_s2(torch.nn.functional.relu(self.linear_s1(x))))[:, 1]

        return policy_x, score_x
