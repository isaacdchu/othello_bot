from random import choice
from typing import Tuple
from src.player import Player
from src.board import Board

class Randobot(Player):
    def __init__(self, name: str, player_id: int) -> None:
        super().__init__(name, player_id)

    def get_move(self, board: Board) -> Tuple[int, int]:
        # Randomly select a legal move
        return choice(list(board.legal_moves))