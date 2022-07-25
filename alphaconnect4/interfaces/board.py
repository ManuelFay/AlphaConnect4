class Board:
    def update_turn(self):
        # TODO: needs to switch every turn to work with current logic. Change that eventually
        # self.turn = 1 - self.turn
        raise NotImplementedError

    def drop_piece(self, action):
        raise NotImplementedError

    def is_valid_action(self, action):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def winning_move(self):
        raise NotImplementedError

    def tie(self):
        """Checks if board is full and score indeterminate
        From the call trace, it should not be possible for the board to be won"""
        raise NotImplementedError

    def get_valid_actions(self):
        """Valid rows to play"""
        raise NotImplementedError
