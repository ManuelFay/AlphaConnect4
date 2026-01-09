import random
from typing import Tuple

from alphaconnect4.engines.mcts import Node
from alphaconnect4.interfaces.backgammon_board import Action, BackgammonBoard


class BackgammonTree(BackgammonBoard, Node):
    def __init__(self, board: BackgammonBoard, dice: Tuple[int, int]):
        super().__init__(
            points=board.points.copy(),
            bar=board.bar.copy(),
            borne_off=board.borne_off.copy(),
            turn=board.turn,
        )
        self.dice = dice
        self.last_move: Action = None
        self._update_id()

    def _update_id(self) -> None:
        """Create a hashable identity for MCTS using board + dice."""
        self.id_ = hash(
            (
                tuple(self.points.tolist()),
                self.bar[0],
                self.bar[1],
                self.borne_off[0],
                self.borne_off[1],
                self.turn,
                self.dice,
            )
        )

    def create_child(self, action: Action) -> "BackgammonTree":
        """Apply an action and roll dice for the next player."""
        child_board = self.copy()
        child_board.apply_action(action)
        child_board.turn = 1 - child_board.turn
        child_dice = child_board.roll_dice()
        child = BackgammonTree(child_board, child_dice)
        child.last_move = action
        return child

    def find_children(self):
        """Expand the node into all legal actions for the current dice."""
        if self.is_terminal():
            return set()

        actions = self.get_legal_actions(self.dice)
        if not actions:
            child_board = self.copy()
            child_board.turn = 1 - child_board.turn
            child = BackgammonTree(child_board, child_board.roll_dice())
            child.last_move = []
            return {child}

        return {self.create_child(action) for action in actions}

    def find_random_child(self):
        """Sample a random child, respecting current dice constraints."""
        if self.is_terminal():
            return None

        actions = self.get_legal_actions(self.dice)
        if not actions:
            child_board = self.copy()
            child_board.turn = 1 - child_board.turn
            child = BackgammonTree(child_board, child_board.roll_dice())
            child.last_move = []
            return child

        action = random.choice(actions)
        return self.create_child(action)

    def reward(self):
        """Return win/loss outcome from the perspective of the current player."""
        if self.borne_off[0] >= 15:
            winner = 0
        elif self.borne_off[1] >= 15:
            winner = 1
        else:
            return 0.5

        return 1.0 if winner == self.turn else 0.0

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
