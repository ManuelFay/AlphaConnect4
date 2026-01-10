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

    def draw_board(
        self,
        board,
        dice,
        possible_actions,
        selected_index,
        last_action=None,
        selected_src=None,
        awaiting_roll=False,
        remaining_dice=None,
        selected_die_index=None,
    ):
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

        if selected_src is not None:
            self._highlight_point(selected_src, board.turn)

        if last_action:
            self._draw_last_action(last_action, board.turn)

        self._draw_info_panel(
            board,
            dice,
            awaiting_roll=awaiting_roll,
            remaining_dice=remaining_dice,
            selected_die_index=selected_die_index,
        )
        self._draw_roll_button()

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
            label_rect = move_label.get_rect()
            label_rect.topleft = (BOARD_MARGIN, self.height - 60)
            pygame.draw.rect(
                self.screen,
                WHITE,
                (label_rect.x - 6, label_rect.y - 4, label_rect.width + 12, label_rect.height + 8),
            )
            pygame.draw.rect(
                self.screen,
                BLUE,
                (label_rect.x - 6, label_rect.y - 4, label_rect.width + 12, label_rect.height + 8),
                2,
            )
            self.screen.blit(move_label, label_rect)

        pygame.display.update()

    def _draw_info_panel(self, board, dice, awaiting_roll=False, remaining_dice=None, selected_die_index=None):
        panel_rect = pygame.Rect(0, 0, self.width, BOARD_MARGIN)
        pygame.draw.rect(self.screen, WHITE, panel_rect)

        player_text = f"Turn: Player {board.turn}"
        turn_label = self.small_font.render(player_text, 1, TEXT_COLOR)
        self.screen.blit(turn_label, (BOARD_MARGIN, 8))

        off_text = f"Off: {board.borne_off[0]} | {board.borne_off[1]}"
        off_label = self.small_font.render(off_text, 1, TEXT_COLOR)
        self.screen.blit(off_label, (self.width - 180, 8))
        self._draw_off_zone()

        dice_start_x = self.width // 2 - DICE_SIZE - 8
        if awaiting_roll:
            roll_text = "Click Roll"
            roll_label = self.small_font.render(roll_text, 1, RED)
            self.screen.blit(roll_label, (dice_start_x - 80, 8))
        self.die_rects = []
        left_value = dice[0]
        right_value = dice[1]
        highlight_left = selected_die_index == 0 if remaining_dice else False
        highlight_right = selected_die_index == 1 if remaining_dice else False
        self._draw_die(dice_start_x, 4, left_value, highlight=highlight_left)
        self._draw_die(dice_start_x + DICE_SIZE + 8, 4, right_value, highlight=highlight_right)

    def _draw_die(self, x, y, value, highlight=False):
        die_rect = pygame.Rect(x, y, DICE_SIZE, DICE_SIZE)
        self.die_rects.append(die_rect)
        pygame.draw.rect(self.screen, BOARD_COLOR, die_rect)
        border_color = BLUE if highlight else BLACK
        pygame.draw.rect(self.screen, border_color, die_rect, 3 if highlight else 2)

        pip_positions = {
            1: [(0.5, 0.5)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
            4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
            5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
            6: [(0.25, 0.2), (0.25, 0.5), (0.25, 0.8), (0.75, 0.2), (0.75, 0.5), (0.75, 0.8)],
        }
        for px, py in pip_positions.get(value, []):
            center = (int(x + DICE_SIZE * px), int(y + DICE_SIZE * py))
            pygame.draw.circle(self.screen, BLACK, center, 4)

    def _draw_roll_button(self):
        self.roll_rect = pygame.Rect(self.width - 130, self.height - 50, 100, 32)
        pygame.draw.rect(self.screen, WHITE, self.roll_rect)
        pygame.draw.rect(self.screen, BLUE, self.roll_rect, 2)
        label = self.small_font.render("Roll", 1, BLUE)
        label_rect = label.get_rect(center=self.roll_rect.center)
        self.screen.blit(label, label_rect)

    def is_roll_clicked(self, pos):
        return hasattr(self, "roll_rect") and self.roll_rect.collidepoint(pos)

    def die_at_pos(self, pos):
        if not hasattr(self, "die_rects"):
            return None
        for index, rect in enumerate(self.die_rects):
            if rect.collidepoint(pos):
                return index
        return None

    def point_at_pos(self, pos):
        x, y = pos
        if y < BOARD_MARGIN or y > self.height - BOARD_MARGIN:
            return None

        for idx in range(24):
            is_top = idx >= 12
            col = idx - 12 if is_top else 11 - idx
            x_start = self._x_positions()[col]
            x_end = x_start + POINT_WIDTH
            if x_start <= x <= x_end:
                if is_top and y <= BOARD_MARGIN + POINT_HEIGHT:
                    return idx
                if (not is_top) and y >= BOARD_MARGIN + POINT_HEIGHT:
                    return idx

        bar_x = BOARD_MARGIN + POINT_WIDTH * 6
        if bar_x <= x <= bar_x + POINT_WIDTH:
            return "bar"

        off_rect = self._off_rect()
        if off_rect.collidepoint(pos):
            return "off"

        return None

    def _off_rect(self):
        return pygame.Rect(self.width - 130, BOARD_MARGIN + 10, 100, 30)

    def _draw_off_zone(self):
        off_rect = self._off_rect()
        pygame.draw.rect(self.screen, WHITE, off_rect)
        pygame.draw.rect(self.screen, BLUE, off_rect, 2)
        label = self.small_font.render("Off", 1, BLUE)
        label_rect = label.get_rect(center=off_rect.center)
        self.screen.blit(label, label_rect)

    def _highlight_point(self, point, turn):
        if point == "bar":
            bar_x = BOARD_MARGIN + POINT_WIDTH * 6
            rect = pygame.Rect(bar_x, BOARD_MARGIN, POINT_WIDTH, POINT_HEIGHT * 2)
        elif point == "off":
            rect = self._off_rect()
        else:
            is_top = point >= 12
            col = point - 12 if is_top else 11 - point
            x = self._x_positions()[col]
            y = BOARD_MARGIN if is_top else BOARD_MARGIN + POINT_HEIGHT
            rect = pygame.Rect(x, y, POINT_WIDTH, POINT_HEIGHT)
        pygame.draw.rect(self.screen, RED if turn == 0 else BLUE, rect, 3)

    def _draw_last_action(self, action, turn):
        if not action:
            return
        move = action[-1]
        src, dest, _ = move
        src_pos = self._point_center(src)
        dest_pos = self._point_center(dest)
        if src_pos and dest_pos:
            last_player = 1 - turn
            color = RED if last_player == 0 else BLUE
            pygame.draw.line(self.screen, color, src_pos, dest_pos, 3)
            self._draw_arrow_head(dest_pos, src_pos, color)

    def _point_center(self, point):
        if point == "bar":
            bar_x = BOARD_MARGIN + POINT_WIDTH * 6 + POINT_WIDTH / 2
            return int(bar_x), int(self.height / 2)
        if point == "off":
            rect = self._off_rect()
            return int(rect.centerx), int(rect.centery)
        if isinstance(point, int):
            is_top = point >= 12
            col = point - 12 if is_top else 11 - point
            x = self._x_positions()[col] + POINT_WIDTH / 2
            y = BOARD_MARGIN + POINT_HEIGHT / 2 if is_top else BOARD_MARGIN + POINT_HEIGHT * 1.5
            return int(x), int(y)
        return None

    def _draw_arrow_head(self, tip, tail, color):
        dx = tip[0] - tail[0]
        dy = tip[1] - tail[1]
        length = max(1, (dx ** 2 + dy ** 2) ** 0.5)
        ux, uy = dx / length, dy / length
        left = (tip[0] - 10 * ux - 5 * uy, tip[1] - 10 * uy + 5 * ux)
        right = (tip[0] - 10 * ux + 5 * uy, tip[1] - 10 * uy - 5 * ux)
        pygame.draw.polygon(self.screen, color, [tip, left, right])
