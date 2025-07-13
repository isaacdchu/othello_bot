"""
Bot that plays Othello.
"""

from __future__ import annotations
from random import choice

from src.board import Board
from src.player import Player

class Otto(Player):
    """
    Otto is a bot that plays Othello.
    It implements the Player interface and provides methods to make moves.
    """

    def __init__(self, name: str, color: str, board: Board):
        super().__init__(name, color, board)

    def get_move(self) -> str:
        """
        Returns a move for the bot based on the current board state.
        This method should implement the bot's strategy for making moves.
        """
        # Placeholder for bot logic (random move for now)
        valid_moves = self.board.get_valid_moves(self.color)
        if not valid_moves:
            return "pass"
        return choice(valid_moves)