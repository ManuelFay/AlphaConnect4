import math
import numpy as np

from engines.base_agent import BaseAgent
from engines.minimax_engine import MinimaxEngine


def sigmoid(x):
    return 1 / (1 + np.exp(-0.05*(x-30)))


class MinimaxAgent(BaseAgent):
    def __init__(self, max_depth: int = 5):
        super().__init__()
        self.max_depth = max_depth

    def move(self, board, turn):
        col, score = MinimaxEngine(board, turn=turn).minimax(self.max_depth, -math.inf, math.inf, True)
        self.ai_confidence = (sigmoid(score)/2) + 0.5
        print(score, self.ai_confidence)
        return col
