from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from src.config_handler import get_config

class Board:
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    def __init__(self):
        # Black's pieces are represented by self.state[0] to begin
        self.state: np.ndarray = Board.__convert_to_state(get_config("config.yaml").board.starting_position)
        # Stores the set of empty squares that neighbor an opponent's piece
        self.potential_moves: Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]
        self.__initialize_potential_moves()
        # Stores the set of legal moves for each player
        self.legal_moves: Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]
        self.__initialize_legal_moves()

    def __is_legal_move(self, move: Tuple[int, int]) -> bool:
        # Assumes that the move is in bounds
        # Check if the move is in the set of potential moves (also ensures that the square is empty)
        if move not in self.potential_moves[0] and move not in self.potential_moves[1]:
            return False
        # Check if the move captures any opponent's pieces
        for di, dj in Board.DIRECTIONS:
            ni, nj = move[0] + di, move[1] + dj
            found_opponent = False
            while 0 <= ni < 8 and 0 <= nj < 8:
                if self.state[1, ni, nj] > 0:
                    found_opponent = True
                    ni += di
                    nj += dj
                    continue
                if self.state[0, ni, nj] > 0:
                    if found_opponent:
                        return True
                    break
                break
        return False

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
            # Print rows with indices (0 at top) and column indices (0-7) at the bottom
            for idx, row in enumerate(board_repr):
                print(f"{idx} " + ' '.join(row))
            col_labels: str = '  ' + ' '.join([str(j) for j in range(8)])
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
    
    def __initialize_potential_moves(self) -> None:
        black_potential: Set[Tuple[int, int]] = set()
        white_potential: Set[Tuple[int, int]] = set()
        all_pieces: np.ndarray = self.state[0] + self.state[1]
        for i in range(8):
            for j in range(8):
                if all_pieces[i, j] != 0:
                    continue
                for di, dj in Board.DIRECTIONS:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= 8 or nj < 0 or nj >= 8:
                        continue
                    if self.state[1, ni, nj] > 0:
                        black_potential.add((i, j))
                    if self.state[0, ni, nj] > 0:
                        white_potential.add((i, j))
        self.potential_moves = (black_potential, white_potential)

    def __initialize_legal_moves(self) -> None:
        black_legal: Set[Tuple[int, int]] = set()
        white_legal: Set[Tuple[int, int]] = set()
        for i, j in self.potential_moves[0]:
            if self.__is_legal_move((i, j)):
                black_legal.add((i, j))
        for i, j in self.potential_moves[1]:
            if self.__is_legal_move((i, j)):
                white_legal.add((i, j))
        self.legal_moves = (black_legal, white_legal)