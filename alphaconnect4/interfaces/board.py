import numpy as np
from scipy.signal import convolve2d

from alphaconnect4.constants.constants import COLUMN_COUNT, ROW_COUNT


class Board:
    def __init__(self, board, turn):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT)).astype(np.uint8) if board is None else board
        self.turn = turn
        self.last_move = None
        self.detection_kernels = [
            np.ones((1, 4), dtype=np.uint8),
            np.ones((4, 1), dtype=np.uint8),
            np.eye(4, dtype=np.uint8),
            np.fliplr(np.eye(4, dtype=np.uint8)),
        ]

        # Possible action set for all turns
        self.action_indices = list(range(self.board.shape[1]))
        assert isinstance(self.board, np.ndarray)

    def update_turn(self):
        self.turn = 1 - self.turn

    def drop_piece(self, row, col):
        self.board[row, col] = self.turn + 1
        self.last_move = col
        self.update_turn()

    def is_valid_location(self, col):
        return self.board[ROW_COUNT - 1, col] == 0

    def get_next_open_row(self, col):
        open_rows = np.where(self.board[:, col] == 0)[0]
        if len(open_rows) > 0:
            return min(open_rows)
        return None

    def __str__(self):
        return np.flip(self.board, 0).tostring()

    def winning_move(self, piece):
        for kernel in self.detection_kernels:
            if (convolve2d(self.board == piece, kernel, mode="valid") == 4).any():
                return True
        return False

    def tie(self):
        """Checks if board is full and score indeterminate
        From the call trace, it should not be possible for the board to be won"""
        return not (self.board == 0).any()

    def get_valid_locations(self):
        """Valid rows to play"""
        return np.where(self.board[-1, :] == 0)[0]
