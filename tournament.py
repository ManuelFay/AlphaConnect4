from dataclasses import dataclass
from random import sample
from typing import Optional

from alphaconnect4.agents.mcts_agent import MCTSAgent
from alphaconnect4.agents.minimax_agent import MinimaxAgent
from alphaconnect4.agents.neural_mcts_agent import NeuralMCTSAgent
from gameplay.game import Game

# Maximum number of rollouts per turn, used for MCTSAgent and NeuralMCTSAgent.
# Set to None for unlimited rollouts (still limited by simulation time).
MAX_ROLLOUTS = 1000


@dataclass
class Player:
    name: str
    type: str
    time: float = 3
    pretrained_path: Optional[str] = None
    rating: int = 1000

    # new fields for variance‐based K
    volatility: float = 1.0  # starts at 1.0, decays towards 0
    decay_rate: float = 0.99  # each game: volatility *= decay_rate


def instanciator(player: Player):
    if player.type == "mcts":
        return MCTSAgent(simulation_time=player.time, max_rollouts=MAX_ROLLOUTS)
    if player.type == "neural_mcts":
        return NeuralMCTSAgent(
            simulation_time=player.time, model_path=player.pretrained_path, max_rollouts=MAX_ROLLOUTS
        )
    if player.type == "minimax":
        return MinimaxAgent(max_depth=int(player.time))
    raise ValueError


contestants = [
    Player("p1", "mcts", time=3),
    Player("p2", "neural_mcts", time=3, pretrained_path="./models/model_naive0.pth"),
    # Player("p3", "minimax", time=4),
    # Player("p4", "neural_mcts", time=5, pretrained_path="./models/model_random.pth"),
    # Player("p5", "neural_mcts", time=5, pretrained_path="./models/model_transformer0.pth"),
    Player("p6", "neural_mcts", time=3, pretrained_path="./models/model_transformer0.pth"),
]
wins = {p.name: 0 for p in contestants}


def update_elo_with_variance(p0: Player, p1: Player, score0: float, score1: float, base_k: int = 32):
    """
    Updates p0.rating and p1.rating in place, scaling the K‐factor
    by each player's current volatility, then decaying that volatility.
    """
    # 1. Compute expected scores
    exp0 = 1 / (1 + 10 ** ((p1.rating - p0.rating) / 400))
    exp1 = 1 / (1 + 10 ** ((p0.rating - p1.rating) / 400))

    # 2. Effective K‐factors
    k0 = base_k * p0.volatility
    k1 = base_k * p1.volatility

    # 3. Rating adjustments
    delta0 = k0 * (score0 - exp0)
    delta1 = k1 * (score1 - exp1)

    # 4. Apply and round
    p0.rating += int(round(delta0))
    p1.rating += int(round(delta1))

    # 5. Decay volatility
    p0.volatility *= p0.decay_rate
    p1.volatility *= p1.decay_rate


for _ in range(200):
    players = sample(contestants, k=2)
    print(f"Game between {players[0].name} and {players[1].name}")
    game = Game(agent0=instanciator(players[0]), agent1=instanciator(players[1]), enable_ui=False)
    result = game.play()

    print(f"{players[result].name if result != 0.5 else 'Draw - No one'} wins")

    wins[players[0].name] += 1 - result
    wins[players[1].name] += result

    # update elo ratings
    update_elo_with_variance(players[0], players[1], 1 - result, result, base_k=32)

    print(wins)

    print([(player.name, player.rating) for player in contestants])
