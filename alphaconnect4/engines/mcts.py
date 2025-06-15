"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.q_value = defaultdict(float)  # total reward of each node
        self.visit_count = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

        self.unexplored_backlog = []

    def score(self, n):
        if self.visit_count[n] == 0:
            return float("-inf")  # avoid unseen moves
        return int(self.q_value[n]) / self.visit_count[n]  # average reward

    def get_policy(self, node, return_dict=True):
        visit_count = max(1, self.visit_count[node] - 1)
        if return_dict:
            return {n.last_move: self.visit_count[n] / visit_count for n in self.children[node]}
        return [self.visit_count[n] / visit_count for n in sorted(self.children[node], key=lambda x: x.last_move)]

    def choose_deprecated(self, node):
        """Deprecated version of choose - Used for debugging
        Choose the best successor of node. (Choose a move in the game)"""
        # if node.is_terminal():
        #     raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        return max(self.children[node], key=self.score)

    def choose(self, node):
        """Choose the best successor of node. (Choose a move in the game)
        Modified"""
        # if node.is_terminal():
        #     raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        return max(self.children[node], key=lambda x: self.visit_count[x])

    def choose_stochastic(self, node, temperature: float = 0.5):
        """Sample from the policy instead of choosing the max (to generate training samples)
        Temperature controls the degree of exploration and could be adjusted throughout the game"""
        # if node.is_terminal():
        #     raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()  # find_heuristic_child()

        # Sample from the policy instead of choosing the max (to generate training samples)
        childs = list(self.children[node])

        visit_count = np.sum([self.visit_count[n] ** (1 / temperature) for n in childs])
        weights = [(self.visit_count[n] ** (1 / temperature)) / visit_count for n in childs]
        return random.choices(childs, weights=weights).pop()

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"

        path = []
        keys = set(self.children.keys())
        while True:
            if self.unexplored_backlog:
                return self.unexplored_backlog.pop()

            path.append(node)
            if node not in keys or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - keys

            if unexplored:
                for n in unexplored:
                    new_path = path.copy()
                    new_path.append(n)
                    self.unexplored_backlog.append(new_path)
                continue

            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    @staticmethod
    def _simulate(node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.visit_count[node] += 1
            self.q_value[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # assert all(n in self.children for n in self.children[node])

        log_n_vertex = math.log(self.visit_count[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.q_value[n] / self.visit_count[n] + self.exploration_weight * math.sqrt(
                log_n_vertex / self.visit_count[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        raise NotImplementedError

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, node2):
        "Nodes must be comparable"
        raise NotImplementedError
