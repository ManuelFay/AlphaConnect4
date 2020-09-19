import pygame

from gameplay.constants import BLACK, BLUE, SQUARESIZE, COLUMN_COUNT, ROW_COUNT, RED, YELLOW, RADIUS


class VisualEngine:
    def __init__(self):
        pygame.init()
        self.width = COLUMN_COUNT * SQUARESIZE
        self.height = (ROW_COUNT + 1) * SQUARESIZE
        size = (self.width, self.height)
        self.screen = pygame.display.set_mode(size)
        self.myfont = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board):
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE,
                                 (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == 1:
                    pygame.draw.circle(self.screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), self.height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
                elif board[r][c] == 2:
                    pygame.draw.circle(self.screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), self.height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
        pygame.display.update()