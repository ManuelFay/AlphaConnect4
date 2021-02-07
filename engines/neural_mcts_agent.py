from engines.mcts_agent import MCTSAgent
from engines.neural_mcts import NeuralMCTS


class NeuralMCTSAgent(MCTSAgent):
    def __init__(self,
                 simulation_time: float = 3.,
                 tree_path: str = None,
                 is_training: bool = False,
                 show_pbar=False,
                 model_path=None):
        super().__init__(simulation_time, tree_path, is_training, show_pbar)
        self.tree = NeuralMCTS(model_path=model_path)

    # def estimate_confidence(self, board):
    #     """Confidence estimation assuming optimal adversary"""
    #
    #     optimal_board = self.tree.choose(board)
    #     if not optimal_board.is_terminal():
    #         score, _ = self.tree.neural_interface.score(optimal_board)
    #         score = 1 - score
    #     else:
    #         score, _ = self.tree.neural_interface.score(board)
    #
    #     return score
