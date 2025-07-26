"""
Interface for a player in the Othello game.
"""

from abc import abstractmethod
from typing import Tuple
from src.board import Board

class Player:
    def __init__(self, name: str, player_id: int) -> None:
        """
        Initializes a player with a given name.
        
        Args:
            name (str): The name of the player.
            player_id (int): The ID of the player (0 or 1).
        """
        self.name = name
        self.player_id = player_id # 0 for black, 1 for white

    def __str__(self) -> str:
        """
        Returns the string representation of the player.
        
        Returns:
            str: The name of the player.
        """
        return self.name

    @abstractmethod
    def get_move(self, board: Board) -> Tuple[int, int]:
        """
        Gets the player's move based on the current state of the board.
        
        Args:
            board (Board): The current state of the game board.
        
        Returns:
            Tuple[int, int]: The coordinates of the move (row, column).
        """
        ...