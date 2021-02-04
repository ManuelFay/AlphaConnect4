import sys
from tqdm import tqdm

from gameplay.game import Game

from engines.mcts_agent import MCTSAgent
from engines.neural_mcts_agent import NeuralMCTSAgent

for _ in tqdm(range(20)):
    agent0 = MCTSAgent(simulation_time=1, tree_path=None, is_training=True)
    agent1 = MCTSAgent(simulation_time=1, tree_path=None, is_training=True)

    game = Game(agent0=agent0, agent1=agent1, enable_ui=False)
    game.play()

sys.exit()
