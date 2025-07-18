"""
Interface for a player in the Othello game.
"""

from typing import Tuple
from src.board import Board

class Player:
    def __init__(self, name: str) -> None:
        """
        Initializes a player with a given name.
        
        Args:
            name (str): The name of the player.
        """
        self.name = name

    def __str__(self) -> str:
        """
        Returns the string representation of the player.
        
        Returns:
            str: The name of the player.
        """
        return self.name
    
    def get_move(self, board: Board) -> Tuple[int, int]:
        """
        Gets the player's move based on the current state of the board.
        
        Args:
            board (Board): The current state of the game board.
        
        Returns:
            Tuple[int, int]: The coordinates of the move (row, column).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")