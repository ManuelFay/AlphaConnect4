import random

from alphaconnect4.constants.constants import AI_PIECE, PLAYER_PIECE
from alphaconnect4.engines.mcts import Node
from alphaconnect4.interfaces.board import Board


class Connect4Tree(Board, Node):
    def __init__(self, board, turn):
        self.id_ = None
        super().__init__(board, turn)
        self.update_id()

    def create_child(self, row, col):
        child = Connect4Tree(self.board.copy(), turn=self.turn)
        child.drop_piece(row, col)
        child.update_id()
        return child

    def update_id(self):
        self.id_ = hash(self.board.tobytes())

    def is_terminal(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or self.tie()

    def find_children(self):
        if self.is_terminal():  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        childs = set()

        for col in self.get_valid_locations():
            row = self.get_next_open_row(col)
            childs.add(self.create_child(row, col))

        return childs

    def find_random_child(self):
        if self.is_terminal():
            return None  # If the game is finished then no moves can be made

        col = random.choice(self.get_valid_locations())
        row = self.get_next_open_row(col)
        return self.create_child(row, col)

    def reward(self):
        # Remove failsafes
        # if not self.is_terminal():
        #     raise RuntimeError(f"reward called on non-terminal board {self}")
        #
        # if (self.winning_move(piece=2) and (self.turn == 1)) or (self.winning_move(piece=1) and (self.turn == 0)):
        #     # It's your turn and you've already won. Should be impossible.
        #     raise RuntimeError(f"reward called on unreachable board {self}")
        # if (self.winning_move(piece=1) and (self.turn == 1)) or (self.winning_move(piece=2) and (self.turn == 0)):
        #     return 0  # Your opponent has just won. Bad.
        # if len(self.get_valid_locations()) == 0:
        #     return 0.5  # Board is a tie
        # # The winner is neither True, False, nor None
        # raise RuntimeError("board has unknown winner type")

        return 0.5 if len(self.get_valid_locations()) == 0 else 0

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
