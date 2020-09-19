import numpy as np

from gameplay.constants import ROW_COUNT, COLUMN_COUNT


class Board:
    def __init__(self, board=None, turn=0):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT)).astype(np.uint8) if board is None else board
        self.turn = turn
        self.last_move = None
        assert isinstance(self.board, np.ndarray)

    def update_turn(self):
        self.turn = 0 if self.turn else 1

    def drop_piece(self, row, col):
        self.board[row][col] = self.turn + 1
        self.last_move = col
        self.update_turn()

    def is_valid_location(self, col):
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for row in range(ROW_COUNT):
            if self.board[row][col] == 0:
                return row
        return None

    def __str__(self):
        return str(np.flip(self.board, 0))

    def winning_move(self, piece):
        # Check horizontal locations for win
        for col in range(COLUMN_COUNT - 3):
            for row in range(ROW_COUNT):
                if self.board[row][col] == piece and self.board[row][col + 1] == piece and self.board[row][
                    col + 2] == piece and \
                        self.board[row][
                            col + 3] == piece:
                    return True

        # Check vertical locations for win
        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT - 3):
                if self.board[row][col] == piece and self.board[row + 1][col] == piece and self.board[row + 2][
                    col] == piece and \
                        self.board[row + 3][
                            col] == piece:
                    return True

        # Check positively sloped diagonals
        for col in range(COLUMN_COUNT - 3):
            for row in range(ROW_COUNT - 3):
                if self.board[row][col] == piece and self.board[row + 1][col + 1] == piece and self.board[row + 2][
                    col + 2] == piece and \
                        self.board[row + 3][
                            col + 3] == piece:
                    return True

        # Check negatively sloped diagonals
        for col in range(COLUMN_COUNT - 3):
            for row in range(3, ROW_COUNT):
                if self.board[row][col] == piece and self.board[row - 1][col + 1] == piece and self.board[row - 2][
                    col + 2] == piece and \
                        self.board[row - 3][
                            col + 3] == piece:
                    return True
        return False

    def get_valid_locations(self):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations
