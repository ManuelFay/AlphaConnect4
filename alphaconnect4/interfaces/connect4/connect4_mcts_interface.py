import random
from alphaconnect4.engines.mcts import Node
from alphaconnect4.interfaces.connect4.connect4_board import Connect4Board


class Connect4Tree(Connect4Board, Node):
    def __init__(self, board, turn):
        self.id_ = None
        super().__init__(board, turn)
        self.update_id()

    def create_child(self, action):
        child = self.__class__(self.board.copy(), turn=self.turn)
        child.drop_piece(action)
        child.update_id()
        return child

    def update_id(self):
        self.id_ = hash(self.board.tostring())

    def is_terminal(self):
        return self.winning_move() or self.tie()

    def find_children(self):
        if self.is_terminal():  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        childs = set()

        for action in self.get_valid_actions():
            childs.add(self.create_child(action))

        return childs

    def find_random_child(self):
        if self.is_terminal():
            return None  # If the game is finished then no moves can be made

        row, col = random.choice(self.get_valid_actions())
        return self.create_child(row, col)

    def reward(self):
        return 0.5 if len(self.get_valid_actions()) == 0 else 0

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
