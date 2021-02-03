import time
from tqdm import tqdm

from engines.mcts_agent import MCTSAgent
from engines.neural_mcts import NeuralMCTS

from engines.mcts_interface import Connect4Tree


class NeuralMCTSAgent(MCTSAgent):
    def __init__(self, simulation_time: float = 3., tree_path: str = None, is_training: bool = False):
        super().__init__(simulation_time, tree_path, is_training)
        self.tree = NeuralMCTS()
