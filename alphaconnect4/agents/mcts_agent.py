# pylint: disable=too-many-instance-attributes

import os
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.engines.mcts import MCTS
from alphaconnect4.interfaces.mcts_interface import Connect4Tree


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        simulation_time: float = 3.0,
        max_rollouts: Optional[int] = None,
        training_path: Optional[str] = None,
        show_pbar: bool = False,
    ):
        """is_training: weakens the agent to get more diverse training samples"""
        super().__init__()
        self.simulation_time = simulation_time
        self.max_rollouts = max_rollouts if max_rollouts is not None else int(simulation_time * 10000)
        self.tree = MCTS()
        self.training_path = training_path
        self.is_training = training_path is not None
        self.show_pbar = show_pbar and (not self.is_training)

        if self.is_training:
            self.boards = []
            self.policies = []

    def save_state(self, board):
        tmp_policy = self.tree.get_policy(board, return_dict=True)
        policy = [tmp_policy.get(action, 0) for action in board.action_indices]

        board_ = board.board.copy()

        # Flip board so that agent always has pieces #1
        if board.turn == 1:
            board_[board.board == 1] = 2
            board_[board.board == 2] = 1

        self.policies.append(policy)
        self.boards.append(board_)

    def estimate_confidence(self, board):
        """Confidence estimation assuming optimal adversary"""
        optimal_board = self.tree.choose(board)
        return self.tree.score(optimal_board)

    def move(self, board, turn):
        board = Connect4Tree(board, turn=turn)

        num_rollout = 0
        timeout_start = time.time()
        if self.show_pbar:
            pbar = tqdm()
        while time.time() < timeout_start + self.simulation_time and num_rollout < self.max_rollouts:
            num_rollout += 1
            self.tree.do_rollout(board)
            if self.show_pbar:
                pbar.update()
        self.tree.unexplored_backlog = []

        if self.is_training:
            self.save_state(board)

        if (board.board != 0).sum() < 10 and self.is_training:
            optimal_board = self.tree.choose_stochastic(board, temperature=0.5)
        else:
            optimal_board = self.tree.choose(board)

        self.ai_confidence = self.estimate_confidence(board)
        return optimal_board.last_move

    def kill_agent(self, result: float):
        """Store learning samples"""
        if self.is_training:
            training_samples = np.array([self.boards, self.policies, [result] * len(self.boards)], dtype=object)

            if os.path.isfile(self.training_path):
                train_ = np.load(self.training_path, allow_pickle=True)
                train_ = np.hstack((train_, training_samples))
            else:
                train_ = training_samples

            np.save(self.training_path, train_)
