# pylint: disable=no-member
import sys
import math
import random

import pygame

from alphaconnect4.interfaces.quarto import QuartoBoard, QuartoTree
from alphaconnect4.interfaces.quarto.constants import YELLOW, RED, BLUE, SQUARESIZE, RADIUS, BLACK, WHITE
from alphaconnect4.agents.base_agent import BaseAgent

from gameplay.quarto.visual_engine import VisualEngine


class Game:
    def __init__(self, agent0: BaseAgent = None, agent1: BaseAgent = None, enable_ui: bool = True):
        self.board = QuartoBoard(board=None, turn=random.choice([0, 1]))
        self.game_over = False
        self.agent0 = agent0
        self.agent1 = agent1

        self.result = None
        self.visual_engine = None

        if enable_ui:
            self.visual_engine = VisualEngine()
            self.visual_engine.draw_board(self.board.board)

    def make_move(self, action):
        if self.board.is_valid_action(action):
            self.board.drop_piece(action)

            if self.board.winning_move():
                self.board.update_turn()
                if self.visual_engine:
                    label = self.visual_engine.myfont.render(f"Player {self.board.turn} wins!!", 1,
                                                             YELLOW if self.board.turn else RED)
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
        """ Game routine - call the visual engine, the UI, the AI and the board state."""
        i = 0
        posx, posy = None, None
        while not self.game_over:
            default_move = self.board.available_pieces[i % len(self.board.available_pieces)]

            if self.board.turn == 0 and self.agent0 is not None:  # If it is the AI turn
                board = QuartoTree(board=self.board.board, turn=self.board.turn)
                action = self.agent0.move(board)
                self.make_move(action)
                if self.visual_engine:
                    print(f"Agent 0 Confidence: {self.agent0.ai_confidence}")
                    self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                continue

            if self.board.turn == 1 and self.agent1 is not None:  # If it is the AI turn
                board = QuartoTree(board=self.board.board, turn=self.board.turn)
                action = self.agent1.move(board)
                self.make_move(action)
                if self.visual_engine:
                    print(f"Agent 1 Confidence: {self.agent1.ai_confidence}")
                    self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                continue

            if self.visual_engine:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                        # Ask for Player n Input
                        posx, posy = event.pos[0], event.pos[1]
                        col, row = int(math.floor(posx / SQUARESIZE)), int(math.floor((self.visual_engine.height - posy)/ SQUARESIZE))
                        # self.make_move((row, col, random.choice(self.board.available_pieces)))
                        self.make_move((row, col, default_move))
                        self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)

                    if event.type == pygame.KEYDOWN:
                        i += -1 if event.scancode == 80 else 1

                        default_move = self.board.available_pieces[i % len(self.board.available_pieces)]
                        self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                        if posx and posy:
                            self.visual_engine.draw_piece((posx, posy), default_move)

                    if event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                        posx, posy = event.pos[0], event.pos[1]
                        self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
                        self.visual_engine.draw_piece((posx, posy), default_move)

        if self.visual_engine:
            self.visual_engine.draw_board(self.board.board, self.agent1.ai_confidence if self.agent1 else 0)
            pygame.time.wait(3000)

        if self.agent0:
            self.agent0.kill_agent(result=self.result if self.result == 0.5 else int(self.result == 0))
        if self.agent1:
            self.agent1.kill_agent(result=self.result if self.result == 0.5 else int(self.result == 1))

        return self.result
