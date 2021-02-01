import sys
from gameplay.game import Game

from engines.mcts_agent import MCTSAgent
from engines.minimax_agent import MinimaxAgent

# Setup players (None is a human player, MCTSAgent, MinimaxAgent)
agent0 = None
agent1 = MCTSAgent(simulation_time=3, tree_path=None)
# agent1 = MinimaxAgent(max_depth=5)

game = Game(agent0=None, agent1=agent1)
game.play()
sys.exit()
