"""
Human player implementation for the Othello game.
"""

from typing import Tuple
from src.player import Player
from src.board import Board

class Human(Player):
    def __init__(self, name: str, player_id: int) -> None:
        """
        Initializes a human player with a given name.
        """
        super().__init__(name, player_id)

    def get_move(self, board: Board) -> Tuple[int, int]:
        """
        Gets the player's move based on the current state of the board.

        Args:
            board (Board): The current state of the game board.

        Returns:
            Tuple[int, int]: The coordinates of the move (row, column).
        """
        while True:
            try:
                move = input(f"{self.name}, enter your move (row col): ").strip()
                row, col = map(int, move.split())
                if (row, col) in board.legal_moves:
                    return row, col
                else:
                    print("Invalid move. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter row and column numbers.")