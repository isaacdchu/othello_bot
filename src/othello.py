from __future__ import annotations
from typing import List, Optional, Tuple

from src.config_handler import get_config

class Board:
    def __init__(self):
        self.grid: List[List[Optional[str]]] = get_config('config.yaml').board.starting_position

    def __str__(self):
        return '\n'.join([' '.join(['.' if cell is None else cell for cell in row]) for row in self.grid])
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Board):
            return False
        return self.grid == value.grid
    
    def print_board(self):
        """
        Prints the current state of the board with coordinates.
        """
        footer = '  ' + ' '.join(chr(ord('a') + i) for i in range(8))
        rows = []
        for i, row in enumerate(self.grid):
            row_str = ' '.join(['.' if cell is None else cell for cell in row])
            rows.append(f"{8 - i} {row_str}")
        return '\n'.join(rows) + '\n' + footer
    
    def game_result(self) -> Tuple[str, int, int]:
        """
        Determines the game result based on the current board state.
        Returns 'B' for Black win, 'W' for White win, or 'D' for draw.
        """
        black_count = sum(cell == 'B' for row in self.grid for cell in row)
        white_count = sum(cell == 'W' for row in self.grid for cell in row)

        winner = 'D'  # Default to draw
        if black_count > white_count:
            winner = 'B'
        elif white_count > black_count:
            winner = 'W'
        return winner, black_count, white_count

    def can_move(self, player: str) -> bool:
        """
        Checks if the player can make any valid moves.
        Returns True if there are valid moves, False otherwise.
        """
        return any(self._verify_move(player, row, col) for row in range(8) for col in range(8))

    def make_move(self, player: str, row: int, col: int) -> bool:
        """
        Modifies the board by placing a player's piece at the specified coordinates.
        Returns True if the move is valid and made, False otherwise.
        """
        if not self._verify_move(player, row, col):
            return False
        self.grid[row][col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            self._flip_in_direction(player, row, col, dx, dy)
        return True

    def _flip_in_direction(self, player: str, x: int, y: int, dx: int, dy: int) -> None:
        opponent = 'B' if player == 'W' else 'W'
        nx, ny = x + dx, y + dy
        if not self._is_within_bounds(nx, ny):
            return
        if self.grid[nx][ny] == player:
            return
        while self._is_within_bounds(nx, ny) and self.grid[nx][ny] == opponent:
            nx += dx
            ny += dy
        if not self._is_within_bounds(nx, ny) or self.grid[nx][ny] != player:
            return
        while (nx, ny) != (x + dx, y + dy):
            nx -= dx
            ny -= dy
            self.grid[nx][ny] = player

    def _verify_move(self, player: str, x: int, y: int) -> bool:
        """
        Method for verifying if a move is valid.
        This method should implement the game rules to check if the move can be made.
        """
        if not self._is_valid_player(player):
            return False
        if not self._is_within_bounds(x, y):
            return False
        if not self._is_cell_empty(x, y):
            return False
        
        opponent = 'B' if player == 'W' else 'W'
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        if not self._is_adjacent_to_opponent(x, y, opponent, directions):
            return False
        if not self._captures_opponent(x, y, player, opponent, directions):
            return False

        return True

    def _is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    def _is_valid_player(self, player: str) -> bool:
        return player in ['W', 'B']

    def _is_cell_empty(self, x: int, y: int) -> bool:
        return self.grid[x][y] is None

    def _is_adjacent_to_opponent(self, x: int, y: int, opponent: str, directions: List[tuple]) -> bool:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_within_bounds(nx, ny) and self.grid[nx][ny] == opponent:
                return True
        return False

    def _captures_opponent(self, x: int, y: int, player: str, opponent: str, directions: List[tuple]) -> bool:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not self._is_within_bounds(nx, ny):
                continue
            if self.grid[nx][ny] != opponent:
                continue
            step_x, step_y = nx + dx, ny + dy
            while self._is_within_bounds(step_x, step_y):
                if self.grid[step_x][step_y] is None:
                    break
                if self.grid[step_x][step_y] == player:
                    return True
                step_x += dx
                step_y += dy
        return False