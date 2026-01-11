import sys

from alphaconnect4.agents import BackgammonMCTSAgent
from gameplay.backgammon_game import BackgammonGame

agent0 = None
agent1 = BackgammonMCTSAgent(simulation_time=10.0, show_pbar=True)


game = BackgammonGame(agent0=agent0, agent1=agent1, enable_ui=True)
result = game.play()

if result == 0:
    print(f"{agent0.__class__.__name__ if agent0 else 'Player'} 0 wins !")
elif result == 1:
    print(f"{agent1.__class__.__name__ if agent1 else 'Player'} 1 wins !")
else:
    print("It's a tie")

sys.exit()
