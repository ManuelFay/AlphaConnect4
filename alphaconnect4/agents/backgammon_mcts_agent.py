import time
from typing import Tuple

from tqdm import tqdm

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.engines.mcts import MCTS
from alphaconnect4.interfaces.backgammon_board import Action, BackgammonBoard
from alphaconnect4.interfaces.backgammon_mcts_interface import BackgammonTree


class BackgammonMCTSAgent(BaseAgent):
    def __init__(
        self,
        simulation_time: float = 2.0,
        max_rollouts: int = None,
        show_pbar: bool = False,
    ):
        super().__init__()
        self.simulation_time = simulation_time
        self.max_rollouts = max_rollouts if max_rollouts is not None else int(simulation_time * 2000)
        self.tree = MCTS()
        self.show_pbar = show_pbar

    def move(self, board: BackgammonBoard, dice: Tuple[int, int]) -> Action:
        root = BackgammonTree(board, dice)

        num_rollout = 0
        timeout_start = time.time()
        pbar = tqdm() if self.show_pbar else None
        while time.time() < timeout_start + self.simulation_time and num_rollout < self.max_rollouts:
            num_rollout += 1
            self.tree.do_rollout(root)
            if pbar:
                pbar.update()

        if pbar:
            pbar.close()
        self.tree.unexplored_backlog = []

        best = self.tree.choose(root)
        self.ai_confidence = self.tree.score(best)
        return best.last_move if best else []

    def kill_agent(self, result: float):
        return
