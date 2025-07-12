from __future__ import annotations
from typing import List, Optional

from src.config_handler import get_config

class Board:
    def __init__(self):
        self.grid: List[List[Optional[str]]] = get_config('config.yaml').board.starting_position

    def __str__(self):
        return '\n'.join([' '.join(['.' if cell is None else cell for cell in row]) for row in self.grid])
    
    def make_move(self, player: str, x: int, y: int) -> bool:
        """
        Modifies the board by placing a player's piece at the specified coordinates.
        Returns True if the move is valid and made, False otherwise.
        """
        if not self._verify_move(player, x, y):
            return False
        self.grid[x][y] = player
        return True

    def _verify_move(self, player: str, x: int, y: int) -> bool:
        """
        Method for verifying if a move is valid.
        This method should implement the game rules to check if the move can be made.
        """
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return False
        if player not in ['W', 'B']:
            return False
        if self.grid[x][y] is not None:
            return False
        # Additional game rules for verifying the move would go here.
        return True