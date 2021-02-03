from collections import defaultdict
import math
import random

from engines.mcts import MCTS
from neural_evaluator.neural_interface import NeuralInterface


class NeuralMCTS(MCTS):
    "Neural Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        super().__init__(exploration_weight)
        self.p_value = defaultdict(int)  # total visit count for each node
        self.neural_interface = NeuralInterface()

    def choose_stochastic(self, node):
        """Sample from the policy instead of choosing the max (to generate training samples)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()     # find_heuristic_child()

        # Sample from the policy instead of choosing the max (to generate training samples)
        childrens = list(self.children[node])
        return random.choices(childrens, weights=[self.score(n) for n in childrens]).pop()

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)
        In the neural version, """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        if leaf.is_terminal():
            reward = 1 - leaf.reward()
        else:
            # Verify turn if should be inversed
            score, policy = self.neural_interface.score(leaf)
            reward = 1 - score

            # to add to each children: self.p_value[node] += policy
            for p, n in zip(policy, sorted(self.children[leaf], key=lambda x: x.last_move)):
                self.p_value[n] = p
        self._backpropagate(path, reward)

    def _uct_select(self, node):
        # TODO: check if exploration needs to be more balanced now that we add a factor to it
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_n_vertex = math.log(self.visit_count[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.q_value[n] / self.visit_count[n] + self.exploration_weight * self.p_value[n] * math.sqrt(
                log_n_vertex / self.visit_count[n]
            )

        return max(self.children[node], key=uct)
