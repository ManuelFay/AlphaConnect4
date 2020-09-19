import math
import random
from gameplay.constants import *

from gameplay.board import Board


class MinimaxEngine(Board):
    @staticmethod
    def evaluate_window(window, piece):
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
        score = 0

        ## Score center column
        center_array = [int(i) for i in list(self.board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(self.board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score posiive sloped diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def minimax(self, depth, alpha, beta, maximizingPlayer):
        print(depth)
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node()
        if depth == 0:
            return (None, self.score_position(AI_PIECE))
        if is_terminal:
            if self.winning_move(AI_PIECE):
                return (None, 100000000000000)
            elif self.winning_move(PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)

        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)
                if row is None:
                    pass
                child = MinimaxEngine(self.board.copy())
                child.drop_piece(row, col, AI_PIECE)
                _, new_score = child.minimax(depth-1, alpha, beta, False)
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:   # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)

                if row is None:
                    pass
                child = MinimaxEngine(self.board.copy())
                child.drop_piece(row, col, PLAYER_PIECE)
                _, new_score = child.minimax(depth - 1, alpha, beta, True)
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations
