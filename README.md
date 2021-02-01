# RL-Based Connect Four - A naive implementation


Work in progress repo in which the final goal is to implement a Connect-4 AI that trains through self-play and uses
Monte-Carlo Tree Search, and a neural policy and value estimator neural network. Currently, only the MCTS is implemented and is
enough to beat a strong human player with a few seconds of rollout.

A Minimax algorithm (credits ```KeithGalli```) is also presented as one of the options but does not scale well when confronted with too much depth
and is exploitable with a bit of strategy.

![Demo picture](pictures/demo.png)

### Setup

 - In a new virtual environment, run:

```pip install -r dev_requirements.txt```
 
- (Optional) Change game parameters in `gameplay/constants.py`
- Modify ```run_game.py``` to setup the game mode and difficulty

### Parameters

The parameters give control over:

- Number of rows and columns
- Color Scheme + UI parameters
  
In ```run_game.py```, it is possible to choose:

- One player or two player mode
- Type of engine (Monte-Carlo Tree Search / Minimax)
- Strength of the engine (adjust simulation time / depth)
- An option to precompute and store openings (tree_path) for the MCTS engine

### Interactive play

Once game play constants have been defined, run  ```python run_game.py``` to get started 
with play.

In MCTS play (recommended), the white bar on the right indicates the estimated win
percentage of the player 0. In Minimax play, the bar represents the inverse of a heuristic estimation 
of the position strength of player 1.

Enjoy ! PRs welcomed ! 