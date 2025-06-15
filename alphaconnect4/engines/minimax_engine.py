import math
import random

from alphaconnect4.constants.constants import AI_PIECE, COLUMN_COUNT, EMPTY, PLAYER_PIECE, ROW_COUNT, WINDOW_LENGTH
from alphaconnect4.interfaces.board import Board


class MinimaxEngine(Board):
    """Minimax implementation - Recursive programming of a DFS.
    Deepest state scoring is evaluated by a handcrafted heuristic."""

    def is_terminal_node(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def minimax(self, depth, alpha, beta, adversarial):
        """Minimax algorithm - Depth First Search"""

        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node()
        if depth == 0:
            return None, self.score_position(AI_PIECE)
        if is_terminal:
            return None, (
                1000000000000000 * self.winning_move(AI_PIECE) + -1000000000000000 * self.winning_move(PLAYER_PIECE)
            )

        value = -math.inf if adversarial else math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = self.get_next_open_row(col)
            child = MinimaxEngine(self.board.copy(), turn=self.turn)
            child.drop_piece(row, col)
            _, new_score = child.minimax(depth - 1, alpha, beta, not adversarial)

            if adversarial:
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
            else:
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

    @staticmethod
    def evaluate_window(window, piece):
        """Evaluate a row, column, or diagonal"""

        score = 0
        opp_piece = PLAYER_PIECE
        if piece == PLAYER_PIECE:
            opp_piece = AI_PIECE

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def score_position(self, piece):
        """Evaluate a board position - given handcrafted heuristic"""
        score = 0

        # Score center column
        center_array = [int(i) for i in list(self.board[:, COLUMN_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for row in range(ROW_COUNT):
            row_array = [int(i) for i in list(self.board[row, :])]
            for col in range(COLUMN_COUNT - 3):
                window = row_array[col : col + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for col in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:, col])]
            for row in range(ROW_COUNT - 3):
                window = col_array[row : row + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                window = [self.board[row + i][col + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                window = [self.board[row + 3 - i][col + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score
