import torch

from neural_evaluator.stub_nn import StubNet
from gameplay.constants import ROW_COUNT, COLUMN_COUNT


class NeuralInterface:
    def __init__(self):
        self.model = StubNet(num_rows=ROW_COUNT, num_cols=COLUMN_COUNT)
        self.softmax = torch.nn.Softmax(dim=0)

    def score(self, node):
        """Should include logic for who's turn it is and get based on that
        Board should be one-hot encoded / categorical"""
        input_ = torch.from_numpy(node.board).float()
        col_evaluation, pos_evaluation = self.model(input_)

        score = self.softmax(pos_evaluation)[1].item()
        policy = self.softmax(col_evaluation).detach().cpu().numpy()
        return score, policy
