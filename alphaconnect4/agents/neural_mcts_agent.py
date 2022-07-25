from typing import Optional

from alphaconnect4.agents.mcts_agent import MCTSAgent
from alphaconnect4.engines.neural_mcts import NeuralMCTS
from alphaconnect4.interfaces.neural_interface import NeuralInterface


class NeuralMCTSAgent(MCTSAgent):
    def __init__(self,
                 simulation_time: float = 3.,
                 training_path: Optional[str] = None,
                 show_pbar: bool = False,
                 neural_interface: Optional[NeuralInterface] = None):
        super().__init__(simulation_time, training_path, show_pbar)
        if neural_interface:
            self.tree = NeuralMCTS(neural_interface=neural_interface)
        else:
            raise AttributeError("Missing neural interface")
