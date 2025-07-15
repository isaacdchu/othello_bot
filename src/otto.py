"""
Bot that plays Othello.
"""

from __future__ import annotations
from typing import Dict
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

    def minimax(self, board: Board, depth: int, maximizing: bool) -> int:
        if depth == 0 or not board.get_valid_moves():
            return board.get_evaluation()
        if maximizing:
            max_eval = -64
            for move in board.get_valid_moves():
                hypothetical_board = board.hypothetical_move(move)
                evaluation = self.minimax(hypothetical_board, depth - 1, False)
                max_eval = max(max_eval, evaluation)
            return max_eval
        else:
            min_eval = 64
            for move in board.get_valid_moves():
                hypothetical_board = board.hypothetical_move(move)
                evaluation = self.minimax(hypothetical_board, depth - 1, True)
                min_eval = min(min_eval, evaluation)
            return min_eval

    def get_move(self) -> str:
        """
        Returns a move for the bot based on the current board state.
        Uses minimax search to look ahead 'depth' moves.
        """
        valid_moves = self.board.get_valid_moves()
        options: Dict[str, int] = {}
        maximizing = self.color == 'B'
        for move in valid_moves:
            hypothetical_board = self.board.hypothetical_move(move)
            options[move] = self.minimax(hypothetical_board, 5, not maximizing)
        print(f"Otto's ({self.color}) options: {options}")
        if not options:
            raise ValueError(f"No valid moves for color {self.color}")
        if maximizing:
            best_move = max(options, key=lambda move: options[move])
        else:
            best_move = min(options, key=lambda move: options[move])
        return best_move