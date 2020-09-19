import numpy as np

from gameplay.constants import ROW_COUNT, COLUMN_COUNT

class Board:
    def __init__(self, board=None):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT)) if board is None else board
        assert type(self.board) == np.ndarray

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def __str__(self):
        return str(np.flip(self.board, 0))

    def winning_move(self, piece):
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and \
                        self.board[r][
                            c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and \
                        self.board[r + 3][
                            c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][
                    c + 2] == piece and \
                        self.board[r + 3][
                            c + 3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][
                    c + 2] == piece and \
                        self.board[r - 3][
                            c + 3] == piece:
                    return True
