# pylint: disable=not-callable, no-member, no-name-in-module

import torch

from alphaconnect4.constants.constants import COLUMN_COUNT, ROW_COUNT
from alphaconnect4.interfaces.naive_nn import NaiveNet
from alphaconnect4.interfaces.transformer_nn import ConvTransformerNet


class NeuralInterface:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            # print(f"Loading weights from {model_path}")
            for architecture in [ConvTransformerNet, NaiveNet]:
                try:
                    self.model = architecture(ROW_COUNT, COLUMN_COUNT)
                    self.model.load_state_dict(torch.load(model_path))
                    print(f"Successfully loaded model from {model_path}")
                    break
                except Exception:
                    print(f"Error loading model with {architecture.__name__}, trying next architecture")
        self.model.eval()

    def score(self, node):
        """Flip board so that agent is always with pieces #1
        Score is from the POV of the next to play"""
        board = node.board.copy()
        tmp_boards = torch.tensor(board)
        input_ = torch.zeros(2, *tmp_boards.shape, dtype=torch.float32)
        input_[0, tmp_boards == 1] = 1
        input_[1, tmp_boards == 2] = 1

        if node.turn == 1:
            input_ = input_[[1, 0], :]

        col_evaluation, score_evaluation = self.model(input_.unsqueeze(0))
        score = score_evaluation.squeeze().item()
        policy = col_evaluation.squeeze().detach().cpu().numpy()
        return score, policy
