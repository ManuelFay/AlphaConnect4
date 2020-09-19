import pygame

from gameplay.constants import BLACK, BLUE, SQUARESIZE, COLUMN_COUNT, ROW_COUNT, RED, YELLOW, RADIUS


class VisualEngine:
    def __init__(self):
        pygame.init()   # pylint: disable=no-member
        self.width = COLUMN_COUNT * SQUARESIZE
        self.height = (ROW_COUNT + 1) * SQUARESIZE
        size = (self.width, self.height)
        self.screen = pygame.display.set_mode(size)
        self.myfont = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board):
        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE,
                                 (col * SQUARESIZE, row * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (
                    int(col * SQUARESIZE + SQUARESIZE / 2), int(row * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)

        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT):
                if board[row][col] == 1:
                    pygame.draw.circle(self.screen, RED, (
                        int(col * SQUARESIZE + SQUARESIZE / 2), self.height - int(row * SQUARESIZE + SQUARESIZE / 2)),
                                       RADIUS)
                elif board[row][col] == 2:
                    pygame.draw.circle(self.screen, YELLOW, (
                        int(col * SQUARESIZE + SQUARESIZE / 2), self.height - int(row * SQUARESIZE + SQUARESIZE / 2)),
                                       RADIUS)
        pygame.display.update()
