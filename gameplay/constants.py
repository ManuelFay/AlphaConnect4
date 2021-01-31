# Game options

# Type of engine (minimax / mcts / 2_players)
AI_TYPE = "mcts"

# AI Computation time for mcts (in sec)
MAX_ROLLOUT = 3
# MAX_DEPTH (for minimax only)
MAX_DEPTH = 7

# Save tree search results for future reference  for the first SAVE_MOVES moves
SAVE_MOVES = 0


# Constants

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (150, 150, 150)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)
