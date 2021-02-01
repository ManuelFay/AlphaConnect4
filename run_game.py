import sys
from gameplay.game import Game

from engines.mcts_agent import MCTSAgent
from engines.minimax_agent import MinimaxAgent

# Setup players (None is a human player, MCTSAgent, MinimaxAgent)

agent0 = None
# agent0 = MinimaxAgent(max_depth=5, is_agent1=False)
# agent0 = MCTSAgent(simulation_time=1, tree_path=None)

agent1 = MCTSAgent(simulation_time=3, tree_path=None)
# agent1 = MinimaxAgent(max_depth=5, is_agent1=True)

game = Game(agent0=agent0, agent1=agent1)
result = game.play()

if result == 0:
    print(f"{agent0.__class__.__name__ if agent0 else 'Player'} 0 wins !")
elif result == 1:
    print(f"{agent1.__class__.__name__ if agent1 else 'Player'} 1 wins !")
else:
    print("It's a tie")

sys.exit()
