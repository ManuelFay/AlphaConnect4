from typing import Optional

from alphaconnect4.agents.mcts_agent import MCTSAgent
from alphaconnect4.engines.neural_mcts import NeuralMCTS


class NeuralMCTSAgent(MCTSAgent):
    def __init__(
        self,
        simulation_time: float = 3.0,
        max_rollouts: Optional[int] = None,
        training_path: Optional[str] = None,
        show_pbar: bool = False,
        model_path: Optional[str] = None,
    ):
        super().__init__(simulation_time, max_rollouts, training_path, show_pbar)
        self.tree = NeuralMCTS(model_path=model_path)

    def estimate_confidence(self, board):
        """Confidence estimation assuming optimal adversary"""
        optimal_board = self.tree.choose(board)
        mcts_score = self.tree.score(optimal_board)
        # neural_score, _ = self.tree.neural_interface.score(board)
        # print(f"Neural score: {neural_score},  MCTS score: {mcts_score}")
        return mcts_score
