from typing import Optional
from engines.mcts_agent import MCTSAgent
from engines.neural_mcts import NeuralMCTS


class NeuralMCTSAgent(MCTSAgent):
    def __init__(self,
                 simulation_time: float = 3.,
                 training_path: bool = None,
                 show_pbar: bool = False,
                 model_path: Optional[str] = None):
        super().__init__(simulation_time, training_path, show_pbar)
        self.tree = NeuralMCTS(model_path=model_path)
