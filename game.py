# pylint: disable=no-member
import sys
import math
import random

import pygame

from gameplay.board import Board
from gameplay.visual_engine import VisualEngine
from gameplay.constants import YELLOW, RED, MAX_DEPTH, AI_ON, SQUARESIZE, RADIUS, BLACK

from engines.minimax_engine import MinimaxEngine


class Game:
    def __init__(self):
        self.board = Board()
        self.turn = random.choice([0, 1])
        self.game_over = False

        print(self.board)

        self.visual_engine = VisualEngine()
        self.visual_engine.draw_board(self.board.board)

        self.play()

    def update_turn(self):
        self.turn = 0 if self.turn else 1

    def make_move(self, col):
        if self.board.is_valid_location(col):
            row = self.board.get_next_open_row(col)
            self.board.drop_piece(row, col, self.turn + 1)

            if self.board.winning_move(self.turn + 1):
                label = self.visual_engine.myfont.render(f"Player {self.turn + 1} wins!!", 1,
                                                         YELLOW if self.turn else RED)
                self.visual_engine.screen.blit(label, (40, 10))
                self.game_over = True

    def ai_move(self):
        """Naive AI method"""
        col, score = MinimaxEngine(self.board.board).minimax(MAX_DEPTH, -math.inf, math.inf, True)
        print(f"Score {score}")
        return col

    def play(self):
        while not self.game_over:

            if self.turn == 1 and AI_ON:  # If it is the AI turn

                col = self.ai_move()

                self.make_move(col)
                print(self.board)
                self.visual_engine.draw_board(self.board.board)
                self.update_turn()
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                    posx = event.pos[0]
                    pygame.draw.circle(self.visual_engine.screen, YELLOW if self.turn else RED,
                                       (posx, int(SQUARESIZE / 2)), RADIUS)

                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                    # Ask for Player n Input
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    self.make_move(col)
                    print(self.board)
                    self.visual_engine.draw_board(self.board.board)
                    self.update_turn()


Game()
pygame.time.wait(3000)
