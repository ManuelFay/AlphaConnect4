import random
from engines.mcts import Node
from engines.minimax_engine import MinimaxEngine
from gameplay.constants import PLAYER_PIECE, AI_PIECE

from gameplay.board import Board


class Connect4Tree(Board, Node):

    def is_terminal(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def find_children(self):
        if self.is_terminal():  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        childs = set()

        for col in self.get_valid_locations():
            row = self.get_next_open_row(col)
            child = Connect4Tree(self.board.copy(), turn=self.turn)
            child.drop_piece(row, col)
            childs.add(child)

        return childs

    def find_random_child(self):
        if self.is_terminal():
            return None  # If the game is finished then no moves can be made

        col = random.choice(self.get_valid_locations())
        row = self.get_next_open_row(col)
        child = Connect4Tree(self.board.copy(), turn=self.turn)
        child.drop_piece(row, col)
        return child

    def find_heuristic_child(self):

        if self.is_terminal():
            return None  # If the game is finished then no moves can be made

        child = max(self.find_children(), key=lambda x: MinimaxEngine(x.board, turn=x.turn).score_position(x.turn + 1))
        child = Connect4Tree(child.board.copy(), turn=child.turn)

        return child

    def reward(self):
        if not self.is_terminal():
            raise RuntimeError(f"reward called on nonterminal board {self}")

        if (self.winning_move(piece=2) and (self.turn == 1)) or (self.winning_move(piece=1) and (self.turn == 0)):
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self}")
        if (self.winning_move(piece=1) and (self.turn == 1)) or (self.winning_move(piece=2) and (self.turn == 0)):
            return 0  # Your opponent has just won. Bad.
        if len(self.get_valid_locations()) == 0:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError("board has unknown winner type")

    def __hash__(self):
        return hash(self.board.tostring())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
