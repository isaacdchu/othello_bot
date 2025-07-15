"""
Othello game board implementation.
This module defines the Board class, which represents the Othello game board,
including methods for making moves, checking valid moves, and determining the game result.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from src.config_handler import get_config

class Board:
    # All 8 directions for Othello: diagonals, horizontals, and verticals
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    def __init__(self, board: Optional[Board] = None):
        if board is not None:
            self.grid = [row[:] for row in board.grid]
            self.moves_made = board.moves_made.copy()
            self.open_squares = board.open_squares.copy()
            self.current_player = board.current_player  # <-- FIX: copy the current player!
            self.game_over = board.game_over
        else:
            self.grid: List[List[Optional[str]]] = get_config('config.yaml').board.starting_position
            self.moves_made: Dict[str, None] = {'d4': None, 'e4': None, 'd5': None, 'e5': None}
            self.open_squares: Dict[str, None] = {Board._convert_to_move(row, col): None for row in range(8) for col in range(8) if self.grid[row][col] is None}
            self.current_player: str = 'B'
            self.game_over = False  # Flag to indicate if the game is over

    def __str__(self):
        return '\n'.join([' '.join(['.' if cell is None else cell for cell in row]) for row in self.grid])
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Board):
            return False
        return self.grid == value.grid

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.grid))
    
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
    
    def get_valid_moves(self) -> List[str]:
        """
        Returns a list of valid moves for the current player.
        Each move is represented as a string in the format 'a1', 'b2', etc.
        """
        valid_moves: List[str] = []
        for move in self.open_squares.keys():
            row, col = Board._convert_to_coordinates(move)
            if self._verify_move(row, col):
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

    def can_move(self) -> bool:
        """
        Checks if the current player can make any valid moves.
        Returns True if there are valid moves, False otherwise.
        """
        return any(self._verify_move(row, col) for row in range(8) for col in range(8))

    def make_move(self, row: int, col: int) -> bool:
        """
        Modifies the board by placing a player's piece at the specified coordinates.
        Returns True if the move is valid and made, False otherwise.
        """
        if not self._verify_move(row, col):
            return False
        self.grid[row][col] = self.current_player
        for dx, dy in self.DIRECTIONS:
            self._flip_in_direction(row, col, dx, dy)
        self.moves_made[Board._convert_to_move(row, col)] = None
        self.open_squares.pop(Board._convert_to_move(row, col), None)
        self._switch_player()
        if self._detect_game_over():
            self.game_over = True
            return True
        if not self.can_move():
            self._switch_player()
        return True

    def hypothetical_move(self, move: str) -> Board:
        """
        Returns a new Board instance with the hypothetical move applied.
        This does not modify the current board.
        """
        row, col = Board._convert_to_coordinates(move)
        if not self.can_move():
            raise ValueError(f"No valid moves for color {self.current_player}")
        if not self._verify_move(row, col):
            raise ValueError(f"Invalid move: {move} for color {self.current_player}")
        new_board = Board(self)
        new_board.make_move(row, col)
        return new_board
    
    def get_evaluation(self) -> int:
        """
        Evaluates the board state (positive means black is winning)
        """
        scores: Tuple[int, int] = self.get_player_scores()
        return scores[0] - scores[1]
    
    def _detect_game_over(self) -> bool:
        """
        Checks if the game has ended. Does not modify the board.
        The game ends when neither player can make a valid move.
        Note: This method is used to determine if the game is over without changing the current state.
        """
        if len(self.moves_made) == 64:  # All squares filled
            return True
        # Check if current player can make a move
        if self.can_move():
            return False
        # Check if other player can make a move
        self._switch_player()
        if self.can_move():
            self._switch_player()
            return False
        # If neither player can move, the game is over
        self._switch_player()
        return True
    
    def _switch_player(self) -> None:
        """
        Switches the current player.
        """
        self.current_player = 'W' if self.current_player == 'B' else 'B'

    def _flip_in_direction(self, x: int, y: int, dx: int, dy: int) -> None:
        opponent = 'B' if self.current_player == 'W' else 'W'
        nx, ny = x + dx, y + dy
        if not self._is_within_bounds(nx, ny):
            return
        if self.grid[nx][ny] == self.current_player:
            return
        while self._is_within_bounds(nx, ny) and self.grid[nx][ny] == opponent:
            nx += dx
            ny += dy
        if not self._is_within_bounds(nx, ny) or self.grid[nx][ny] != self.current_player:
            return
        while (nx, ny) != (x + dx, y + dy):
            nx -= dx
            ny -= dy
            self.grid[nx][ny] = self.current_player

    def _verify_move(self, x: int, y: int) -> bool:
        """
        Method for verifying if a move is valid.
        """
        if not self._is_within_bounds(x, y):
            return False
        if not self._is_cell_empty(x, y):
            return False
        if not self._is_adjacent_to_opponent(x, y):
            return False
        if not self._captures_opponent(x, y):
            return False

        return True

    def _is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    def _is_cell_empty(self, x: int, y: int) -> bool:
        return self.grid[x][y] is None

    def _is_adjacent_to_opponent(self, x: int, y: int) -> bool:
        opponent = 'B' if self.current_player == 'W' else 'W'
        for dx, dy in self.DIRECTIONS:
            nx, ny = x + dx, y + dy
            if self._is_within_bounds(nx, ny) and self.grid[nx][ny] == opponent:
                return True
        return False

    def _captures_opponent(self, x: int, y: int) -> bool:
        opponent = 'B' if self.current_player == 'W' else 'W'
        for dx, dy in self.DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not self._is_within_bounds(nx, ny):
                continue
            if self.grid[nx][ny] != opponent:
                continue
            step_x, step_y = nx + dx, ny + dy
            while self._is_within_bounds(step_x, step_y):
                if self.grid[step_x][step_y] is None:
                    break
                if self.grid[step_x][step_y] == self.current_player:
                    return True
                step_x += dx
                step_y += dy
        return False