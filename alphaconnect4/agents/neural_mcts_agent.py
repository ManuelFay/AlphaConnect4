from typing import Optional

from alphaconnect4.agents.mcts_agent import MCTSAgent
from alphaconnect4.engines.neural_mcts import NeuralMCTS


class NeuralMCTSAgent(MCTSAgent):
    def __init__(
        self,
        simulation_time: float = 3.0,
        training_path: Optional[str] = None,
        show_pbar: bool = False,
        model_path: Optional[str] = None,
    ):
        super().__init__(simulation_time, training_path, show_pbar)
        self.tree = NeuralMCTS(model_path=model_path)

    def estimate_confidence(self, board):
        """Confidence estimation assuming optimal adversary"""
        optimal_board = self.tree.choose(board)
        neural_score, _ = self.tree.neural_interface.score(optimal_board)
        mcts_score = self.tree.score(optimal_board)

        print(f"Neural score: {1 - neural_score},  MCTS score: {mcts_score}")
        return mcts_score
