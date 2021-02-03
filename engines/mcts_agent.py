import os
import time
import pickle
from tqdm import tqdm

import numpy as np
from engines.base_agent import BaseAgent
from engines.mcts import MCTS
from engines.mcts_interface import Connect4Tree


class MCTSAgent(BaseAgent):
    def __init__(self, simulation_time: float = 3., tree_path: str = None, is_training: bool = False):
        """is_training: weakens the agent to get more diverse training samples"""
        super().__init__()
        self.simulation_time = simulation_time
        self.tree_path = tree_path
        self.tree = MCTS()
        self.is_training = is_training

        if tree_path and os.path.isfile(tree_path):
            # Load precomputed MC Tree
            with open(tree_path, "rb") as file:
                self.tree = pickle.load(file)

        if self.is_training:
            self.boards = []
            self.policies = []

    def save_state(self, board):
        policy = self.tree.get_policy(board)
        board_ = board.board.copy()

        # Flip board so that agent always has pieces #1
        if board.turn == 1:
            # Would be better just to switch dimensions around when we will have 2 layers
            board_[board.board == 1] = 2
            board_[board.board == 2] = 1

        self.policies.append(policy)
        self.boards.append(board_)

    def estimate_confidence(self, board):
        """Confidence estimation assuming optimal adversary"""
        # self.ai_confidence = self.tree.score(self.tree.choose(board))
        optimal_board = self.tree.choose(board)
        if not optimal_board.is_terminal():
            return 1 - self.tree.score(self.tree.choose(optimal_board))
        else:
            return self.tree.score(optimal_board)

    def move(self, board, turn):
        board = Connect4Tree(board, turn=turn)

        timeout_start = time.time()
        pbar = tqdm()
        while time.time() < timeout_start + self.simulation_time:
            self.tree.do_rollout(board)
            # TODO Async if we want dynamic confidence updates
            # if self.tree.visit_count[board] > 200 and self.tree.visit_count[board] % 10 == 0:
            #     self.ai_confidence = self.estimate_confidence(board)
            #     self.visual_engine.draw_board(board, self.ai_confidence)
            pbar.update()

        if self.is_training:
            optimal_board = self.tree.choose_stochastic(board)
            self.save_state(board)
        else:
            optimal_board = self.tree.choose(board)

        col = optimal_board.last_move
        self.ai_confidence = self.estimate_confidence(board)
        print(f"AI Confidence: {self.ai_confidence}")
        return col

    def save_tree(self):
        # Save new tree exploration info
        if self.tree_path and os.path.isfile(self.tree_path):
            with open(self.tree_path, "wb") as file:
                pickle.dump(self.tree, file)

    def kill_agent(self, result: float):
        """Store learning samples"""
        self.save_tree()
        if self.is_training:
            training_samples = np.array([self.boards, self.policies, [result]*len(self.boards)])
            # Should be in append mode
            # np.save("training.npy", training_samples)
