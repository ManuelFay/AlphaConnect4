# RL-Based Connect Four - A naive implementation


Work in progress repo in which the final goal is to implement a Connect 4 AI that trains through self-play and uses
Monte-Carlo Tree Search, and a neural policy and value estimator neural network. Currently, only the MCTS is implemented and is
enough to beat a strong human player with a few seconds of rollout.

A Minimax algorithm is also presented as one of the options but does not scale well when confronted with too much depth
and is exploitable with a bit of strategy.

![Demo picture](pictures/demo.png)

### Setup

 - In a new virtual environment, run:

```pip install -r dev_requirements.txt```
 
- Define game parameters in `gameplay/parameters.py`

### Parameters

The parameters give control over:

- Number of rows and columns
- One player or two player mode
- Type of engine in the one player mode (Monte-Carlo Tree Search / Minimax)
- Strength of the engine (adjust rollout time / depth)  
- Color Scheme + UI parameters
- An option to precompute and store openings (LOOKUP_PATH)

### Interactive play

Once game play constants have been defined, run  ```python game.py``` to get started 
with play.

In MCTS play (recommended), the white bar on the right indicates the estimated win
percentage of the player.

Enjoy ! PRs welcomed ! 