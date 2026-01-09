# pylint: disable=no-member
import sys
import time

import pygame

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.interfaces.backgammon_board import BackgammonBoard
from gameplay.backgammon_visual_engine import BackgammonVisualEngine


class BackgammonGame:
    def __init__(self, agent0: BaseAgent = None, agent1: BaseAgent = None, enable_ui: bool = True):
        """Gameplay loop for backgammon with optional AI agents and UI."""
        self.board = BackgammonBoard(turn=0)
        self.agent0 = agent0
        self.agent1 = agent1
        self.game_over = False
        self.result = None
        self.dice = self.board.roll_dice()
        self.awaiting_roll = True
        self.last_action = []
        self.visual_engine = BackgammonVisualEngine() if enable_ui else None

    def _apply_action(self, action):
        """Apply the selected action and advance to the next turn/dice."""
        if action:
            self.board.apply_action(action)
            self.last_action = action
        self.board.turn = 1 - self.board.turn
        self.awaiting_roll = True

    def _check_game_over(self):
        """Set terminal flags when a player bears off all checkers."""
        if self.board.is_terminal():
            self.game_over = True
            if self.board.borne_off[0] >= 15:
                self.result = 0
            else:
                self.result = 1

    def _wait_for_roll(self):
        """Wait for a click on the roll button before rolling dice."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.visual_engine and self.visual_engine.is_roll_clicked(event.pos):
                        self.dice = self.board.roll_dice()
                        self.awaiting_roll = False
                        return

            if self.visual_engine:
                self.visual_engine.draw_board(
                    self.board,
                    self.dice,
                    [],
                    0,
                    last_action=self.last_action,
                    awaiting_roll=self.awaiting_roll,
                )
            time.sleep(0.05)

    def _wait_for_human_move(self, dice_sequence):
        """Handle mouse selection for a human move, one die at a time."""
        action = []
        for die in dice_sequence:
            legal_moves = self.board.legal_single_moves(die)
            if not legal_moves:
                continue

            selected_src = None
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.visual_engine and self.visual_engine.is_roll_clicked(event.pos):
                            continue
                        clicked = self.visual_engine.point_at_pos(event.pos) if self.visual_engine else None
                        if clicked is None:
                            continue
                        if selected_src is None:
                            if any(move[0] == clicked for move in legal_moves):
                                selected_src = clicked
                        else:
                            matching = [
                                move for move in legal_moves if move[0] == selected_src and move[1] == clicked
                            ]
                            if matching:
                                move = matching[0]
                                self.board.apply_single_move(move)
                                action.append(move)
                                selected_src = None
                                break
                            selected_src = None

                if self.visual_engine:
                    display_actions = [self.board.describe_action([move]) for move in legal_moves]
                    self.visual_engine.draw_board(
                        self.board,
                        self.dice,
                        display_actions,
                        0,
                        last_action=self.last_action,
                        selected_src=selected_src,
                    )
                time.sleep(0.05)

        return action

    def play(self):
        """Run the full game loop until completion."""
        while not self.game_over:
            current_agent = self.agent0 if self.board.turn == 0 else self.agent1
            if current_agent is None and self.awaiting_roll:
                self._wait_for_roll()

            if self.awaiting_roll:
                self.dice = self.board.roll_dice()
                self.awaiting_roll = False

            if current_agent is not None:
                action = current_agent.move(self.board.copy(), self.dice)
                self._apply_action(action)
                if self.visual_engine:
                    next_actions = self.board.get_legal_actions(self.dice)
                    display_actions = [self.board.describe_action(a) for a in next_actions]
                    self.visual_engine.draw_board(
                        self.board,
                        self.dice,
                        display_actions,
                        0,
                        last_action=self.last_action,
                    )
            else:
                dice_sequence = (
                    [self.dice[0]] * 4 if self.dice[0] == self.dice[1] else [self.dice[0], self.dice[1]]
                )
                action = self._wait_for_human_move(dice_sequence)
                self._apply_action(action)

            self._check_game_over()

        if self.visual_engine:
            display_actions = []
            self.visual_engine.draw_board(self.board, self.dice, display_actions, 0, last_action=self.last_action)
            pygame.time.wait(3000)

        if self.agent0:
            self.agent0.kill_agent(result=int(self.result == 0))
        if self.agent1:
            self.agent1.kill_agent(result=int(self.result == 1))

        return self.result
