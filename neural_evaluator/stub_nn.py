import torch


class StubNet(torch.nn.Module):
    def __init__(self, num_rows: int, num_cols: int):
        super(StubNet, self).__init__()
        self.linear = torch.nn.Linear(in_features=num_rows*num_cols, out_features=num_cols)
        self.linear2 = torch.nn.Linear(in_features=num_rows*num_cols, out_features=2)

    def forward(self, x):
        x = x.view(-1, self.linear.in_features)
        column_x = self.linear(x)
        score_x = self.linear2(x)
        return column_x, score_x
