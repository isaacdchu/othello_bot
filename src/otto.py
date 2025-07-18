from typing import Tuple
from src.player import Player
from src.board import Board

class Otto(Player):
    def __init__(self, name: str):
        super().__init__(name)

    def get_move(self, board: Board) -> Tuple[int, int]:
        return (0, 0)  # Placeholder for Otto's move logic