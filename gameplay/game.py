# pylint: disable=no-member
import math
import random
import sys

import pygame

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.constants.constants import BLACK, BLUE, RADIUS, RED, SQUARESIZE, YELLOW
from alphaconnect4.interfaces.board import Board
from gameplay.visual_engine import VisualEngine


class Game:
    def __init__(self, agent0: BaseAgent = None, agent1: BaseAgent = None, enable_ui: bool = True):
        self.board = Board(board=None, turn=random.choice([0, 1]))
        self.game_over = False
        self.agent0 = agent0
        self.agent1 = agent1

        self.result = None
        self.visual_engine = None

        if enable_ui:
            self.visual_engine = VisualEngine()
            self.visual_engine.draw_board(self.board.board)

    def make_move(self, col):
        if self.board.is_valid_location(col):
            row = self.board.get_next_open_row(col)
            self.board.drop_piece(row, col)

            if self.board.winning_move((1 - self.board.turn) + 1):
                self.board.update_turn()
                if self.visual_engine:
                    label = self.visual_engine.myfont.render(
                        f"Player {self.board.turn} wins!!", 1, YELLOW if self.board.turn else RED
                    )
                    self.visual_engine.screen.blit(label, (40, 10))
                self.game_over = True
                self.result = self.board.turn

            elif self.board.tie():
                if self.visual_engine:
                    label = self.visual_engine.myfont.render("It's a tie!", 1, BLUE)
                    self.visual_engine.screen.blit(label, (40, 10))
                self.game_over = True
                self.result = 0.5

    def play(self):
        """Game routine - call the visual engine, the UI, the AI and the board state."""
        while not self.game_over:
            if self.board.turn == 0 and self.agent0 is not None:  # If it is the AI turn
                col = self.agent0.move(board=self.board.board, turn=self.board.turn)
                self.make_move(col)
                if self.visual_engine:
                    print(f"Agent 0 Confidence: {self.agent0.ai_confidence}")
                    self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                continue

            if self.board.turn == 1 and self.agent1 is not None:  # If it is the AI turn
                col = self.agent1.move(board=self.board.board, turn=self.board.turn)
                self.make_move(col)
                if self.visual_engine:
                    print(f"Agent 1 Confidence: {self.agent1.ai_confidence}")
                    self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                continue

            if self.visual_engine:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                    if event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                        posx = event.pos[0]
                        pygame.draw.circle(
                            self.visual_engine.screen,
                            YELLOW if self.board.turn else RED,
                            (posx, int(SQUARESIZE / 2)),
                            RADIUS,
                        )

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                        # Ask for Player n Input
                        posx = event.pos[0]
                        col = int(math.floor(posx / SQUARESIZE))

                        self.make_move(col)

                    self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)

        if self.visual_engine:
            self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
            pygame.time.wait(3000)

        if self.agent0:
            self.agent0.kill_agent(result=self.result if self.result == 0.5 else int(self.result == 0))
        if self.agent1:
            self.agent1.kill_agent(result=self.result if self.result == 0.5 else int(self.result == 1))

        return self.result
