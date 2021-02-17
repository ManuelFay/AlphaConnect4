from dataclasses import dataclass
from typing import Optional
from random import sample

from gameplay.game import Game

from engines.mcts_agent import MCTSAgent
from engines.neural_mcts_agent import NeuralMCTSAgent
from engines.minimax_agent import MinimaxAgent

from elo.elo_rating import elo_rating


@dataclass
class Player:
    name: str
    type: str
    time: float = 3
    pretrained_path: Optional[str] = None
    rating: int = 1000


def instanciator(player: Player):
    if player.type == "mcts":
        return MCTSAgent(simulation_time=player.time)
    if player.type == "neural_mcts":
        return NeuralMCTSAgent(simulation_time=player.time, model_path=player.pretrained_path)
    if player.type == "minimax":
        return MinimaxAgent(max_depth=int(player.time))
    raise ValueError


contestants = [Player("p1", "mcts", time=3),
               Player("p2", "neural_mcts", time=3, pretrained_path="/home/manu/perso/RL_Connect4/model_0.pth"),
               # Player("p3", "minimax", time=5),
               # Player("p4", "neural_mcts", time=3),
               # Player("p5", "neural_mcts", time=3,  pretrained_path="/home/manu/perso/RL_Connect4/model_1.pth")
               ]
wins = {p.name: 0 for p in contestants}

for _ in range(100):
    players = sample(contestants, k=2)
    print(f"Game between {players[0].name} and {players[1].name}")
    game = Game(agent0=instanciator(players[0]), agent1=instanciator(players[1]), enable_ui=False)
    result = game.play()

    print(f"{players[result].name if result != 0.5 else 'Draw - No one'} wins")

    wins[players[0].name] += 1 - result
    wins[players[1].name] += result
    print(wins)

    updated_ratings = elo_rating(players[0].rating, players[1].rating, result)
    players[0].rating = updated_ratings[0]
    players[1].rating = updated_ratings[1]

print([(player.name, player.rating) for player in contestants])
