# pylint: disable=no-member
import sys
import time
import pickle
import math
import random

from tqdm import tqdm
import pygame

from gameplay.board import Board
from gameplay.visual_engine import VisualEngine
from gameplay.constants import YELLOW, RED, BLUE, MAX_DEPTH, SQUARESIZE, RADIUS, BLACK, AI_TYPE, MAX_ROLLOUT, SAVE_MOVES

from engines.minimax_engine import MinimaxEngine
from engines.mcts import MCTS
from engines.mcts_interface import Connect4Tree


class Game:
    def __init__(self):
        self.board = Board(board=None, turn=random.choice([0, 1]))
        self.game_over = False
        self.tree = None
        if AI_TYPE == "mcts":
            """
            try:
                # Load precomputed MC Tree
                with open("tree.pickle", "rb") as file:
                    self.tree = pickle.load(file)
            except FileNotFoundError:
                # Recreate from scratch
                self.tree = MCTS()
            """
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
                pbar.update()

            col = self.tree.choose(board).last_move
        else:
            raise NameError
        return col

    def play(self):
        """ Game routine - call the visual engine, the UI, the AI and the board state."""
        while not self.game_over:

            if self.board.turn == 1 and AI_TYPE != "2_players":  # If it is the AI turn

                col = self.ai_move()
                self.make_move(col)

                self.visual_engine.draw_board(self.board.board)

                # Save new tree exploration info
                if self.board.move_number < SAVE_MOVES:
                    with open("tree.pickle", "wb") as file:
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
                    self.visual_engine.draw_board(self.board.board)


game = Game()
game.play()

pygame.time.wait(2000)
