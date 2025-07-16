from __future__ import annotations
from typing import List, Optional
import numpy as np
from src.config_handler import get_config

class Board:
    def __init__(self):
        self.state: np.ndarray = Board.__convert_to_state(get_config("config.yaml").board.starting_position)

    def pretty_print(self, coords: bool = False) -> None:
        board_repr: List[List[str]] = []
        for i in range(8):
            row: List[str] = []
            for j in range(8):
                if np.isclose(self.state[0, i, j], 1.0, rtol=1e-09, atol=1e-09):
                    row.append('B')
                elif np.isclose(self.state[1, i, j], 1.0, rtol=1e-09, atol=1e-09):
                    row.append('W')
                else:
                    row.append('.')
            board_repr.append(row)
        if coords:
            # Print rows with numbers (1 at top) and column letters (a-h) at the bottom
            for idx, row in enumerate(board_repr):
                print(f"{idx + 1} " + ' '.join(row))
            col_labels: str = '  ' + ' '.join([chr(ord('a') + j) for j in range(8)])
            print(col_labels)
        else:
            for row in board_repr:
                print(' '.join(row))

    @staticmethod
    def __convert_to_state(board: List[List[Optional[str]]]) -> np.ndarray:
        black: np.ndarray = np.zeros((8, 8), dtype=np.float32)
        white: np.ndarray = np.zeros((8, 8), dtype=np.float32)
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == 'B':
                    black[i, j] = 1.0
                elif cell == 'W':
                    white[i, j] = 1.0
        state: np.ndarray = np.array([black, white], dtype=np.float32)
        return state