"""
Player class for Othello.
This class represents a player in the Othello game, including methods for making moves and evaluating the game state.
"""

from __future__ import annotations

from src.board import Board

class Player:
    def __init__(self, name: str, color: str, board: Board) -> None:
        if color not in ['B', 'W']:
            raise ValueError("Color must be 'B' for Black or 'W' for White.")
        if not isinstance(board, Board):
            raise TypeError("board must be an instance of Board.")
        if not name:
            raise ValueError("Name cannot be empty.")
        self.name = name
        self.color = color  # 'B' for Black, 'W' for White
        self.board = board
        self.score: int = 0

    def __str__(self) -> str:
        return f"{self.name} ({self.color}) - Score: {self.score}"
    
    def get_move(self) -> str:
        """
        Placeholder for getting a move from the player.
        In a real game, this would involve user input or AI logic.
        """
        raise NotImplementedError("This method should be implemented by subclasses or in the game logic.")