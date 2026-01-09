import pygame

from alphaconnect4.constants.backgammon_constants import (
    BAR_COLOR,
    BOARD_COLOR,
    BOARD_MARGIN,
    CHECKER_RADIUS,
    CHECKER_SPACING,
    DICE_SIZE,
    POINT_DARK,
    POINT_HEIGHT,
    POINT_LIGHT,
    POINT_WIDTH,
    TEXT_COLOR,
    WHITE,
    BLACK,
    BLUE,
    RED,
)


class BackgammonVisualEngine:
    def __init__(self):
        """Initialize the pygame surface and fonts for rendering."""
        pygame.init()  # pylint: disable=no-member
        self.width = BOARD_MARGIN * 2 + POINT_WIDTH * 12 + POINT_WIDTH
        self.height = BOARD_MARGIN * 2 + POINT_HEIGHT * 2
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AlphaBackgammon")
        self.font = pygame.font.SysFont("monospace", 20)
        self.small_font = pygame.font.SysFont("monospace", 16)

    def _x_positions(self):
        """X positions for the 12 points on each side, accounting for the bar."""
        positions = []
        for col in range(12):
            x = BOARD_MARGIN + col * POINT_WIDTH
            if col >= 6:
                x += POINT_WIDTH
            positions.append(x)
        return positions

    def _point_rect(self, col, top):
        """Top-left coordinates for a point triangle."""
        x = self._x_positions()[col]
        y = BOARD_MARGIN if top else BOARD_MARGIN + POINT_HEIGHT
        return x, y

    def draw_board(self, board, dice, possible_actions, selected_index):
        """Draw the board, checkers, dice, and move list."""
        self.screen.fill(BOARD_COLOR)

        bar_x = BOARD_MARGIN + POINT_WIDTH * 6
        pygame.draw.rect(
            self.screen,
            BAR_COLOR,
            (bar_x, BOARD_MARGIN, POINT_WIDTH, POINT_HEIGHT * 2),
        )

        for col in range(12):
            color = POINT_LIGHT if col % 2 == 0 else POINT_DARK
            x, y_top = self._point_rect(col, top=True)
            points_top = [
                (x, y_top),
                (x + POINT_WIDTH, y_top),
                (x + POINT_WIDTH / 2, y_top + POINT_HEIGHT),
            ]
            pygame.draw.polygon(self.screen, color, points_top)

            x, y_bottom = self._point_rect(col, top=False)
            points_bottom = [
                (x, y_bottom + POINT_HEIGHT),
                (x + POINT_WIDTH, y_bottom + POINT_HEIGHT),
                (x + POINT_WIDTH / 2, y_bottom),
            ]
            pygame.draw.polygon(self.screen, color, points_bottom)

        for idx, count in enumerate(board.points):
            if count == 0:
                continue
            is_top = idx >= 12
            col = idx - 12 if is_top else 11 - idx
            x = self._x_positions()[col] + POINT_WIDTH / 2
            checker_color = WHITE if count > 0 else BLACK
            stack_count = abs(count)
            for i in range(stack_count):
                if is_top:
                    y = BOARD_MARGIN + CHECKER_RADIUS + i * (CHECKER_RADIUS * 2 + CHECKER_SPACING)
                else:
                    y = (
                        BOARD_MARGIN
                        + POINT_HEIGHT * 2
                        - CHECKER_RADIUS
                        - i * (CHECKER_RADIUS * 2 + CHECKER_SPACING)
                    )
                pygame.draw.circle(self.screen, checker_color, (int(x), int(y)), CHECKER_RADIUS)

        bar_x_center = bar_x + POINT_WIDTH / 2
        for player, count in board.bar.items():
            if count == 0:
                continue
            checker_color = WHITE if player == 0 else BLACK
            for i in range(count):
                y = BOARD_MARGIN + CHECKER_RADIUS + i * (CHECKER_RADIUS * 2 + CHECKER_SPACING)
                pygame.draw.circle(self.screen, checker_color, (int(bar_x_center), int(y)), CHECKER_RADIUS)

        dice_text = f"Dice: {dice[0]}-{dice[1]}"
        dice_label = self.font.render(dice_text, 1, TEXT_COLOR)
        self.screen.blit(dice_label, (BOARD_MARGIN, 10))

        borne_off_text = f"Off: {board.borne_off[0]} | {board.borne_off[1]}"
        off_label = self.small_font.render(borne_off_text, 1, TEXT_COLOR)
        self.screen.blit(off_label, (self.width - 180, 10))

        if possible_actions:
            action_text = f"Moves: {len(possible_actions)}"
        else:
            action_text = "No legal moves"
        action_label = self.small_font.render(action_text, 1, TEXT_COLOR)
        self.screen.blit(action_label, (BOARD_MARGIN, self.height - 30))

        if possible_actions:
            selected_index = max(0, min(selected_index, len(possible_actions) - 1))
            selection = possible_actions[selected_index]
            move_text = f"Selected: {selection}"
            move_label = self.small_font.render(move_text, 1, BLUE)
            self.screen.blit(move_label, (BOARD_MARGIN, self.height - 55))

        pygame.display.update()
