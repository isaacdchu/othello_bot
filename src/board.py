"""
Othello game board implementation.
This module defines the Board class, which represents the Othello game board,
including methods for making moves, checking valid moves, and determining the game result.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from src.config_handler import get_config

class Board:
    def __init__(self):
        self.grid: List[List[Optional[str]]] = get_config('config.yaml').board.starting_position
        self.moves_made: Dict[str, None] = {'d4': None, 'e4': None, 'd5': None, 'e5': None}
        self.open_squares: Dict[str, None] = {self._convert_to_move(row, col): None for row in range(8) for col in range(8) if self.grid[row][col] is None}

    def __str__(self):
        return '\n'.join([' '.join(['.' if cell is None else cell for cell in row]) for row in self.grid])
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Board):
            return False
        return self.grid == value.grid
    
    @staticmethod
    def _convert_to_coordinates(move: str) -> Tuple[int, int]:
        """
        Converts a move string (e.g., 'a1') to board coordinates (row, col).
        """
        letter, number = move[0], move[1]
        col: int = ord(letter) - ord('a')
        row: int = 8 - int(number)
        return (row, col)

    @staticmethod
    def _convert_to_move(row: int, col: int) -> str:
        """
        Converts board coordinates (row, col) to a move string (e.g., 'a1').
        """
        letter = chr(ord('a') + col)
        number = str(8 - row)
        return f"{letter}{number}"
    
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
    
    def get_valid_moves(self, color: str) -> List[str]:
        """
        Returns a list of valid moves for the given player.
        Each move is represented as a string in the format 'a1', 'b2', etc.
        """
        valid_moves: List[str] = []
        for move in self.open_squares.keys():
            row, col = self._convert_to_coordinates(move)
            if self._verify_move(color, row, col):
                valid_moves.append(move)
        return valid_moves
    
    def get_player_scores(self) -> Tuple[int, int]:
        """
        Returns the scores of both players.
        The first value is the score for player 'B' (Black), and the second is for player 'W' (White).
        """
        black_count = sum(cell == 'B' for row in self.grid for cell in row)
        white_count = sum(cell == 'W' for row in self.grid for cell in row)
        return black_count, white_count

    def can_move(self, color: str) -> bool:
        """
        Checks if the player can make any valid moves.
        Returns True if there are valid moves, False otherwise.
        """
        return any(self._verify_move(color, row, col) for row in range(8) for col in range(8))

    def make_move(self, color: str, row: int, col: int) -> bool:
        """
        Modifies the board by placing a player's piece at the specified coordinates.
        Returns True if the move is valid and made, False otherwise.
        """
        if not self._verify_move(color, row, col):
            return False
        self.grid[row][col] = color
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            self._flip_in_direction(color, row, col, dx, dy)
        return True

    def _flip_in_direction(self, color: str, x: int, y: int, dx: int, dy: int) -> None:
        opponent = 'B' if color == 'W' else 'W'
        nx, ny = x + dx, y + dy
        if not self._is_within_bounds(nx, ny):
            return
        if self.grid[nx][ny] == color:
            return
        while self._is_within_bounds(nx, ny) and self.grid[nx][ny] == opponent:
            nx += dx
            ny += dy
        if not self._is_within_bounds(nx, ny) or self.grid[nx][ny] != color:
            return
        while (nx, ny) != (x + dx, y + dy):
            nx -= dx
            ny -= dy
            self.grid[nx][ny] = color

    def _verify_move(self, color: str, x: int, y: int) -> bool:
        """
        Method for verifying if a move is valid.
        This method should implement the game rules to check if the move can be made.
        """
        if not self._is_valid_player(color):
            return False
        if not self._is_within_bounds(x, y):
            return False
        if not self._is_cell_empty(x, y):
            return False

        opponent = 'B' if color == 'W' else 'W'
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        if not self._is_adjacent_to_opponent(x, y, opponent, directions):
            return False
        if not self._captures_opponent(x, y, color, opponent, directions):
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