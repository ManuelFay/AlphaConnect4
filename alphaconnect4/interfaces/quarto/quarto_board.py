import numpy as np
from scipy.signal import convolve2d
from itertools import product

from ..board import Board


class QuartoBoard(Board):
    def __init__(self, board, turn):
        self.board = np.ones((4, 4)).astype(np.uint8) * 16 if board is None else board
        self.pieces = list(range(16))
        self.available_pieces = sorted(list(set(self.pieces.copy()) - set(self.board.flatten().tolist())))
        self.turn = turn
        self.last_move = None
        self.detection_kernels = [np.ones((1, 4), dtype=np.uint8),
                                  np.ones((4, 1), dtype=np.uint8),
                                  np.eye(4, dtype=np.uint8),
                                  np.fliplr(np.eye(4, dtype=np.uint8))]

        # Possible action set for all turns
        # self.action_indices = list(product(range(self.board.shape[0]), range(self.board.shape[1]), self.pieces))
        assert isinstance(self.board, np.ndarray)

    def update_turn(self):
        self.turn = 1 - self.turn

    def drop_piece(self, action):
        (row, col, piece_id) = action
        self.board[row, col] = piece_id
        self.last_move = action
        self.available_pieces.remove(piece_id)
        self.update_turn()

    def is_valid_action(self, action):
        (row, col, piece_id) = action
        if piece_id not in self.available_pieces:
            return False
        return self.board[row, col] == 16

    def __str__(self):
        return np.flip(self.board, 0).tostring()  # check why the flip

    def winning_move(self):
        board = np.unpackbits(np.expand_dims(self.board, 2), 2, bitorder='little').astype(np.bool)
        for channel in range(4):
            for kernel in self.detection_kernels:
                if (convolve2d((~board[:, :, 4] & (board[:, :, channel] == 0)).astype(np.uint8), kernel,
                               mode="valid") == 4).any():
                    return True
                if (convolve2d((~board[:, :, 4] & (board[:, :, channel] == 1)).astype(np.uint8), kernel,
                               mode="valid") == 4).any():
                    return True
            return False

    def tie(self):
        """Checks if board is full and score indeterminate
        From the call trace, it should not be possible for the board to be won"""
        return not (self.board == 16).any()

    def get_valid_actions(self):
        """Valid rows to play"""
        rows, cols = np.where((self.board == 16))
        return [(*row_col, piece_id) for row_col, piece_id in
                list(product(list(zip(rows, cols)), self.available_pieces))]
