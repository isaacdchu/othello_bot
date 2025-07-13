"""
Human player for Othello game.
"""

from __future__ import annotations

from src.board import Board
from src.player import Player

class Human(Player):
    def __init__(self, name: str, color: str, board: Board):
        super().__init__(name, color, board)

    def get_move(self) -> str:
        """
        Prompts the human player for a move.
        Returns the move string.
        """
        return input(f"Player {self.name}, enter your move (e.g., 'a1'): ").strip().lower()
