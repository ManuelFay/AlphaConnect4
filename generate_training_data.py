import sys
from tqdm import tqdm

from alphaconnect4.agents.neural_mcts_agent import NeuralMCTSAgent
from alphaconnect4.interfaces.connect4 import Connect4NeuralInterface as NI

from gameplay.connect4.game import Game

# Parallelize to generate samples in parallel
for _ in tqdm(range(500)):
    try:
        agent0 = NeuralMCTSAgent(simulation_time=3, neural_interface=NI(model_path="./models/model_0.pth"))
        agent1 = NeuralMCTSAgent(simulation_time=3, neural_interface=NI(model_path="./models/model_1.pth"),
                                 training_path="./data/training_2b.npy")

        game = Game(agent0=agent0, agent1=agent1, enable_ui=False)
        game.play()
    except Exception as exception:
        print(exception)

sys.exit()
