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
        self.visual_engine = BackgammonVisualEngine() if enable_ui else None

    def _apply_action(self, action):
        """Apply the selected action and advance to the next turn/dice."""
        if action:
            self.board.apply_action(action)
        self.board.turn = 1 - self.board.turn
        self.dice = self.board.roll_dice()

    def _check_game_over(self):
        """Set terminal flags when a player bears off all checkers."""
        if self.board.is_terminal():
            self.game_over = True
            if self.board.borne_off[0] >= 15:
                self.result = 0
            else:
                self.result = 1

    def _wait_for_human_move(self, possible_actions):
        """Handle keyboard selection for a human player."""
        selected_index = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        selected_index = (selected_index + 1) % max(1, len(possible_actions))
                    elif event.key == pygame.K_LEFT:
                        selected_index = (selected_index - 1) % max(1, len(possible_actions))
                    elif event.key == pygame.K_RETURN:
                        if possible_actions:
                            return possible_actions[selected_index]
                        return []

            if self.visual_engine:
                display_actions = [self.board.describe_action(action) for action in possible_actions]
                self.visual_engine.draw_board(
                    self.board,
                    self.dice,
                    display_actions,
                    selected_index,
                )
            time.sleep(0.05)

    def play(self):
        """Run the full game loop until completion."""
        while not self.game_over:
            current_agent = self.agent0 if self.board.turn == 0 else self.agent1
            possible_actions = self.board.get_legal_actions(self.dice)

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
                    )
            else:
                action = self._wait_for_human_move(possible_actions)
                self._apply_action(action)

            self._check_game_over()

        if self.visual_engine:
            display_actions = []
            self.visual_engine.draw_board(self.board, self.dice, display_actions, 0)
            pygame.time.wait(3000)

        if self.agent0:
            self.agent0.kill_agent(result=int(self.result == 0))
        if self.agent1:
            self.agent1.kill_agent(result=int(self.result == 1))

        return self.result
