"""
Game of Othello (Reversi) implementation.
This module provides the player input handling and game state management.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from src.board import Board
from src.player import Player

class Othello:
    def __init__(self, player_1: Player, player_2: Player, board: Board):
        self.board = board
        self.players = {
            'B': player_1,
            'W': player_2
        }
        self.current_player = 'B'  # Black starts first

    def __str__(self) -> str:
        return str(self.board)

    def __switch_player(self) -> None:
        self.current_player = 'W' if self.current_player == 'B' else 'B'

    def __parse_move(self, move: str) -> Tuple[int, int]:
        letter, number = move[0], move[1]
        col: int = ord(letter) - ord('a')
        row: int = 8 - int(number)
        return (row, col)
    
    def detect_game_end(self) -> bool:
        return not self.board.can_move('B') and not self.board.can_move('W')
    
    def __update_scores(self) -> None:
        black_count, white_count = self.board.get_player_scores()
        self.players['B'].score = black_count
        self.players['W'].score = white_count

    def get_board(self) -> List[List[Optional[str]]]:
        return self.board.grid

    def new_turn(self, move: str) -> bool:
        """        
        Processes a player's move.
        Returns True if the move was made, False otherwise.
        """
        move = move.lower().strip()[:2]
        if move == 'qu':
            raise KeyboardInterrupt
        if move[0] not in 'abcdefgh' or move[1] not in '12345678':
            print("Invalid move format. Use letters a-h and numbers 1-8.")
            return False
        row, col = self.__parse_move(move)
        if not self.board.make_move(self.current_player, row, col):
            print("Invalid move. Try again.")
            return False
        self.__update_scores()
        print(f"Player {self.current_player} made a move at {move}.")
        self.__switch_player()
        if not self.board.can_move(self.current_player):
            print(f"Player {self.current_player} has no valid moves.")
            self.__switch_player()
        return True