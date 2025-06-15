import math
from collections import defaultdict

from alphaconnect4.engines.mcts import MCTS
from alphaconnect4.interfaces.neural_interface import NeuralInterface


class NeuralMCTS(MCTS):
    "Neural Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, model_path=None):
        super().__init__(exploration_weight)
        self.p_value = defaultdict(int)  # total visit count for each node
        self.neural_interface = NeuralInterface(model_path=model_path)

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)
        In the neural version, score leafs with NN and store policy vectors"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        if leaf.is_terminal():
            reward = 1 - leaf.reward()
        else:
            score, policy = self.neural_interface.score(leaf)
            reward = 1 - score

            for child in self.children[leaf]:
                self.p_value[child] = policy[child.last_move]

        self._backpropagate(path, reward)

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # assert all(n in self.children for n in self.children[node])

        vertex_count = math.sqrt(self.visit_count[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.q_value[n] / self.visit_count[n] + self.exploration_weight * self.p_value[n] * vertex_count / (
                1 + self.visit_count[n]
            )

        return max(self.children[node], key=uct)
