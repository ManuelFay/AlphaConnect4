import time
from tqdm import tqdm

from engines.mcts_agent import MCTSAgent
from engines.neural_mcts import NeuralMCTS

from engines.mcts_interface import Connect4Tree


class NeuralMCTSAgent(MCTSAgent):
    def __init__(self, simulation_time: float = 3., tree_path: str = None, is_training: bool = False):
        super().__init__(simulation_time, tree_path)
        self.tree = NeuralMCTS()
        self.is_training = is_training

    def move(self, board, turn):
        board = Connect4Tree(board, turn=turn)
        timeout_start = time.time()
        pbar = tqdm()
        while time.time() < timeout_start + self.simulation_time:
            self.tree.do_rollout(board)
            pbar.update()

        optimal_board = self.tree.choose(board) if not self.is_training else self.tree.choose_stochastic(board)
        col = optimal_board.last_move
        self.ai_confidence = self.estimate_confidence(board)
        print(f"AI Confidence: {self.ai_confidence}")
        self.save_tree()
        return col
