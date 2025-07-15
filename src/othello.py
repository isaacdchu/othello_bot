"""
Game of Othello (Reversi) implementation.
This module provides the player input handling and game state management.
"""

from __future__ import annotations
from typing import Tuple
from src.board import Board
from src.player import Player

class Othello:
    def __init__(self, player_1: Player, player_2: Player, board: Board):
        self.board = board
        self.players = {
            'B': player_1,
            'W': player_2
        }
        self.__current_player: str = 'B'  # Start with player 'B'

    def __str__(self) -> str:
        return str(self.board)

    def __parse_move(self, move: str) -> Tuple[int, int]:
        letter, number = move[0], move[1]
        col: int = ord(letter) - ord('a')
        row: int = 8 - int(number)
        return (row, col)
    
    def __update_scores(self) -> None:
        black_count, white_count = self.board.get_player_scores()
        self.players['B'].score = black_count
        self.players['W'].score = white_count

    def detect_game_over(self) -> bool:
        return self.board.game_over

    def new_turn(self, move: str) -> bool:
        """        
        Processes a player's move.
        Returns True if the move was made, False otherwise.
        """
        move = move.lower().strip()[:2]
        if move == 'qu':
            raise KeyboardInterrupt
        if move[0] not in 'abcdefgh' or move[1] not in '12345678':
            print(f"Invalid move format. Use letters a-h and numbers 1-8. Received: {move}")
            return False
        row, col = self.__parse_move(move)
        if not self.board.make_move(row, col):
            print("Invalid move. Try again.")
            return False
        self.__update_scores()
        print(f"Player {self.__current_player} made a move at {move}.")
        if not self.board.can_move():
            print(f"Player {self.__current_player} has no valid moves.")
        # Switch to the next player
        self.__current_player = self.board.current_player
        return True