import math

import numpy as np

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.engines.minimax_engine import MinimaxEngine


def sigmoid(x):
    x = np.clip(x, a_min=-1000, a_max=1000)
    return 1 / (1 + np.exp(-0.05 * (x - 30)))


class MinimaxAgent(BaseAgent):
    def __init__(self, max_depth: int = 5, is_agent1: bool = True):
        super().__init__()
        self.max_depth = max_depth
        self.is_agent1 = is_agent1

    def move(self, board, turn):
        col, score = MinimaxEngine(board, turn=turn).minimax(self.max_depth, -math.inf, math.inf, self.is_agent1)
        self.ai_confidence = (sigmoid(score) / 2) + 0.5
        # print(score, self.ai_confidence)
        return col
