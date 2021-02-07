import sys
from tqdm import tqdm

from gameplay.game import Game

from engines.mcts_agent import MCTSAgent
from engines.neural_mcts_agent import NeuralMCTSAgent


# Parallelize to generate samples in parallel
for _ in tqdm(range(200)):
    try:
        agent0 = NeuralMCTSAgent(simulation_time=3, model_path="/home/manu/perso/RL_Connect4/model2.pth", is_training=True)
        agent1 = NeuralMCTSAgent(simulation_time=3, model_path="/home/manu/perso/RL_Connect4/model2.pth", is_training=True)

        game = Game(agent0=agent0, agent1=agent1, enable_ui=False)
        game.play()
    except Exception as e:
        print(e)

sys.exit()
