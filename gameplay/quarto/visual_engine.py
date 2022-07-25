import pygame
import numpy as np

from alphaconnect4.interfaces.quarto.constants import BLACK, BLUE, SQUARESIZE, RED, YELLOW, RADIUS, WHITE


ROW_COUNT, COLUMN_COUNT = 4, 4


class VisualEngine:
    def __init__(self):
        pygame.init()   # pylint: disable=no-member
        self.width = COLUMN_COUNT * SQUARESIZE + int(SQUARESIZE/2)
        self.height = (ROW_COUNT + 1) * SQUARESIZE
        size = (self.width, self.height)
        self.screen = pygame.display.set_mode(size)
        self.myfont = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board, ai_confidence: float = 0.5):
        ai_confidence = max(0., min(ai_confidence, 1.))

        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE,
                                 (col * SQUARESIZE, row * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (
                    int(col * SQUARESIZE + SQUARESIZE / 2), int(row * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)

        board = np.unpackbits(np.expand_dims(board, 2), 2, bitorder='little').astype(np.bool)
        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT):
                if not board[row][col][4]:
                    pygame.draw.circle(self.screen, RED if board[row][col][0] else YELLOW,
                                       (int(col * SQUARESIZE + SQUARESIZE / 4),
                                        self.height - int(row * SQUARESIZE + SQUARESIZE / 4)),
                                       RADIUS/2)
                    pygame.draw.circle(self.screen, RED if board[row][col][1] else YELLOW,
                                       (int(col * SQUARESIZE + SQUARESIZE / 4),
                                        self.height - int(row * SQUARESIZE + 3*SQUARESIZE / 4)),
                                       RADIUS / 2)
                    pygame.draw.circle(self.screen, RED if board[row][col][2] else YELLOW,
                                       (int(col * SQUARESIZE + 3*SQUARESIZE / 4),
                                        self.height - int(row * SQUARESIZE + SQUARESIZE / 4)),
                                       RADIUS / 2)
                    pygame.draw.circle(self.screen, RED if board[row][col][3] else YELLOW,
                                       (int(col * SQUARESIZE + 3*SQUARESIZE / 4),
                                        self.height - int(row * SQUARESIZE + 3*SQUARESIZE / 4)),
                                       RADIUS / 2)
                        # pygame.draw.circle(self.screen, BLACK, (
                        #     int(col * SQUARESIZE + SQUARESIZE / 2),
                        #     self.height - int(row * SQUARESIZE + SQUARESIZE / 2)),
                        #                    RADIUS * ((channel + 1) * 1.05 / 5))
                        #
                        # pygame.draw.circle(self.screen, RED if board[row][col][channel] else YELLOW, (
                        #     int(col * SQUARESIZE + SQUARESIZE / 2), self.height - int(row * SQUARESIZE + SQUARESIZE / 2)),
                        #                    RADIUS * ((channel+1) * 0.98/5))

        # Draw confidence
        pygame.draw.rect(self.screen, WHITE,
                         (COLUMN_COUNT * SQUARESIZE, SQUARESIZE, int(SQUARESIZE/2), ROW_COUNT * SQUARESIZE))
        pygame.draw.rect(self.screen, BLACK,
                         (COLUMN_COUNT * SQUARESIZE, SQUARESIZE, int(SQUARESIZE / 2),
                          int((ROW_COUNT * SQUARESIZE * ai_confidence))))

        for tick_level in range(5):
            pygame.draw.rect(self.screen, BLUE,
                             (COLUMN_COUNT * SQUARESIZE, SQUARESIZE + int(ROW_COUNT * SQUARESIZE * tick_level/4),
                              int(SQUARESIZE / 2),
                              1))

        pygame.display.update()
