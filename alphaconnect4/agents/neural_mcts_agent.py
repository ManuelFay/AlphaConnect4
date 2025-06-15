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
