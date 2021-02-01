# pylint: disable=no-member
import sys
import os
import time
import pickle
import math
import random

from tqdm import tqdm
import pygame

from gameplay.board import Board
from gameplay.visual_engine import VisualEngine
from gameplay.parameters import YELLOW, RED, BLUE, MAX_DEPTH, SQUARESIZE, RADIUS, BLACK, AI_TYPE, MAX_ROLLOUT, \
    SAVE_MOVES, LOOKUP_PATH

from engines.minimax_engine import MinimaxEngine
from engines.mcts import MCTS
from engines.mcts_interface import Connect4Tree


class Game:
    def __init__(self):
        self.board = Board(board=None, turn=random.choice([0, 1]))
        self.game_over = False
        self.tree = None
        self.ai_confidence: float = 0.5
        if AI_TYPE == "mcts" and LOOKUP_PATH and os.path.isfile(LOOKUP_PATH):
            # Load precomputed MC Tree
            with open(LOOKUP_PATH, "rb") as file:
                self.tree = pickle.load(file)
        else:
            self.tree = MCTS()

        # Display the board in terminal
        # print(self.board)

        self.visual_engine = VisualEngine()
        self.visual_engine.draw_board(self.board.board)

    def make_move(self, col):
        if self.board.is_valid_location(col):
            row = self.board.get_next_open_row(col)
            self.board.drop_piece(row, col)

            if self.board.winning_move((1-self.board.turn) + 1):
                self.board.update_turn()
                label = self.visual_engine.myfont.render(f"Player {self.board.turn + 1} wins!!", 1,
                                                         YELLOW if self.board.turn else RED)
                self.visual_engine.screen.blit(label, (40, 10))
                self.game_over = True

            elif self.board.tie():
                label = self.visual_engine.myfont.render("It's a tie!", 1, BLUE)
                self.visual_engine.screen.blit(label, (40, 10))
                self.game_over = True

    def estimate_confidence(self, board):
        """Confidence estimation assuming optimal adversary"""
        # self.ai_confidence = self.tree.score(self.tree.choose(board))
        optimal_board = self.tree.choose(board)
        if not optimal_board.is_terminal():
            return 1 - self.tree.score(self.tree.choose(optimal_board))
        else:
            return self.tree.score(optimal_board)

    def ai_move(self):
        """AI method"""
        if AI_TYPE == "minimax":
            col, score = MinimaxEngine(self.board.board, turn=self.board.turn).minimax(MAX_DEPTH, -math.inf, math.inf,
                                                                                       True)
            print(f"Score {score}")
        elif AI_TYPE == "mcts":
            board = Connect4Tree(self.board.board, turn=self.board.turn)

            timeout_start = time.time()
            pbar = tqdm()
            while time.time() < timeout_start + MAX_ROLLOUT:
                self.tree.do_rollout(board)
                if self.tree.visit_count[board] > 200 and self.tree.visit_count[board] % 10 == 0:
                    self.ai_confidence = self.estimate_confidence(board)
                    self.visual_engine.draw_board(self.board.board, self.ai_confidence)
                pbar.update()

            optimal_board = self.tree.choose(board)
            col = optimal_board.last_move
            self.ai_confidence = self.estimate_confidence(board)
            print(f"AI Confidence: {self.ai_confidence}")
        else:
            raise NameError
        return col

    def play(self):
        """ Game routine - call the visual engine, the UI, the AI and the board state."""
        while not self.game_over:

            if self.board.turn == 1 and AI_TYPE != "2_players":  # If it is the AI turn

                col = self.ai_move()
                self.make_move(col)

                self.visual_engine.draw_board(self.board.board, self.ai_confidence)

                # Save new tree exploration info
                if self.board.move_number < SAVE_MOVES:
                    with open(LOOKUP_PATH, "wb") as file:
                        pickle.dump(self.tree, file)
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                    posx = event.pos[0]
                    pygame.draw.circle(self.visual_engine.screen, YELLOW if self.board.turn else RED,
                                       (posx, int(SQUARESIZE / 2)), RADIUS)

                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(self.visual_engine.screen, BLACK, (0, 0, self.visual_engine.width, SQUARESIZE))
                    # Ask for Player n Input
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    self.make_move(col)
                    # print(self.board)
                    self.visual_engine.draw_board(self.board.board, self.ai_confidence)

        pygame.time.wait(2000)


game = Game()
game.play()
