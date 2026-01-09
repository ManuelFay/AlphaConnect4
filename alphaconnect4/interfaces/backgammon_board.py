import copy
import random
from typing import List, Sequence, Tuple, Union

import numpy as np

Move = Tuple[Union[int, str], Union[int, str], int]
Action = List[Move]


class BackgammonBoard:
    def __init__(
        self,
        points: Sequence[int] = None,
        bar: dict = None,
        borne_off: dict = None,
        turn: int = 0,
    ):
        self.points = np.array(points, dtype=int) if points is not None else self._starting_points()
        self.bar = bar if bar is not None else {0: 0, 1: 0}
        self.borne_off = borne_off if borne_off is not None else {0: 0, 1: 0}
        self.turn = turn

    @staticmethod
    def _starting_points() -> np.ndarray:
        """Return the standard backgammon starting position.

        Positive counts are player 0, negative counts are player 1.
        """
        points = np.zeros(24, dtype=int)
        # Player 0 setup (positive)
        points[23] = 2
        points[12] = 5
        points[7] = 3
        points[5] = 5
        # Player 1 setup (negative)
        points[0] = -2
        points[11] = -5
        points[16] = -3
        points[18] = -5
        return points

    def copy(self):
        return BackgammonBoard(
            points=self.points.copy(),
            bar=copy.deepcopy(self.bar),
            borne_off=copy.deepcopy(self.borne_off),
            turn=self.turn,
        )

    @staticmethod
    def roll_dice() -> Tuple[int, int]:
        """Roll two six-sided dice."""
        return random.randint(1, 6), random.randint(1, 6)

    def _player_sign(self) -> int:
        """Return +1 for player 0, -1 for player 1."""
        return 1 if self.turn == 0 else -1

    def _opponent_sign(self) -> int:
        return -self._player_sign()

    def _home_range(self, player: int):
        """Home board indices for each player."""
        return range(0, 6) if player == 0 else range(18, 24)

    def _outside_home_indices(self, player: int):
        return range(6, 24) if player == 0 else range(0, 18)

    def _all_in_home(self, player: int) -> bool:
        """Check if all checkers are in the home board and none are on the bar."""
        sign = 1 if player == 0 else -1
        if self.bar[player] > 0:
            return False
        return not (self.points[list(self._outside_home_indices(player))] * sign > 0).any()

    def is_terminal(self) -> bool:
        return self.borne_off[0] >= 15 or self.borne_off[1] >= 15

    def _entry_point(self, die: int) -> int:
        """Return the point index used when entering from the bar."""
        if self.turn == 0:
            return 24 - die
        return die - 1

    def _is_blocked(self, dest: int) -> bool:
        """Return True if destination is blocked by 2+ opponent checkers."""
        sign = self._player_sign()
        return self.points[dest] * sign <= -2

    def _can_bear_off(self, start: int, die: int) -> bool:
        """Return True if bearing off from start using die is legal."""
        if not self._all_in_home(self.turn):
            return False

        if self.turn == 0:
            if start - die >= 0:
                return False
            higher_points = list(range(start + 1, 6))
            return not (self.points[higher_points] > 0).any()

        if start + die <= 23:
            return False
        lower_points = list(range(18, start))
        return not (self.points[lower_points] < 0).any()

    def _legal_single_moves(self, die: int) -> List[Move]:
        """List all legal single-checker moves for a given die."""
        sign = self._player_sign()
        moves: List[Move] = []

        if self.bar[self.turn] > 0:
            dest = self._entry_point(die)
            if not self._is_blocked(dest):
                moves.append(("bar", dest, die))
            return moves

        for idx, count in enumerate(self.points):
            if count * sign <= 0:
                continue
            dest = idx - die if self.turn == 0 else idx + die
            if 0 <= dest <= 23:
                if not self._is_blocked(dest):
                    moves.append((idx, dest, die))
            else:
                if self._can_bear_off(idx, die):
                    moves.append((idx, "off", die))

        return moves

    def apply_single_move(self, move: Move) -> None:
        """Apply a single move to the board in-place."""
        src, dest, _die = move
        sign = self._player_sign()

        if src == "bar":
            self.bar[self.turn] -= 1
        else:
            self.points[int(src)] -= sign

        if dest == "off":
            self.borne_off[self.turn] += 1
            return

        dest_index = int(dest)
        if self.points[dest_index] * sign == -1:
            self.points[dest_index] = 0
            self.bar[1 - self.turn] += 1
        self.points[dest_index] += sign

    def apply_action(self, action: Action) -> None:
        """Apply a sequence of moves for a single turn."""
        for move in action:
            self.apply_single_move(move)

    def _generate_sequences(self, dice_order: Sequence[int]) -> List[Action]:
        """Generate legal action sequences for a specific dice order."""
        actions: List[Action] = []

        def backtrack(board: "BackgammonBoard", die_index: int, current: Action):
            if die_index >= len(dice_order):
                actions.append(current.copy())
                return

            die = dice_order[die_index]
            moves = board._legal_single_moves(die)
            if not moves:
                actions.append(current.copy())
                return

            for move in moves:
                next_board = board.copy()
                next_board.apply_single_move(move)
                backtrack(next_board, die_index + 1, current + [move])

        backtrack(self, 0, [])
        return actions

    def get_legal_actions(self, dice: Tuple[int, int]) -> List[Action]:
        """Return all legal action sequences for a given dice roll."""
        die1, die2 = dice
        if die1 == die2:
            dice_orders = [[die1] * 4]
        else:
            dice_orders = [[die1, die2], [die2, die1]]

        sequences: List[Action] = []
        for order in dice_orders:
            sequences.extend(self._generate_sequences(order))

        if not sequences:
            return []

        max_len = max(len(seq) for seq in sequences)
        sequences = [seq for seq in sequences if len(seq) == max_len]

        if die1 != die2 and max_len == 1:
            highest = max(die1, die2)
            if any(seq and seq[0][2] == highest for seq in sequences):
                sequences = [seq for seq in sequences if seq and seq[0][2] == highest]

        return sequences

    def describe_action(self, action: Action) -> str:
        """Human-readable string for an action sequence."""
        parts = []
        for src, dest, die in action:
            src_label = "bar" if src == "bar" else str(int(src) + 1)
            dest_label = "off" if dest == "off" else str(int(dest) + 1)
            parts.append(f"{src_label}->{dest_label}({die})")
        return ", ".join(parts) if parts else "pass"
