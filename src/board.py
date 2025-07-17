from typing import List, Set, Tuple, Optional
import numpy as np
from src.config_handler import get_config

class Board:
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    def __init__(self):
        # Black's pieces are represented by self.state[0] to begin
        self.state: np.ndarray = Board.__convert_to_state(get_config("config.yaml").board.starting_position)
        # Stores the set of all empty squares
        self.empty_squares: Set[Tuple[int, int]] = set()
        self.__initialize_empty_squares()
        # Stores the set of empty squares that neighbor an opponent's piece
        self.potential_moves: Set[Tuple[int, int]] = set()
        self.__update_potential_moves()
        # Stores the set of legal moves for the current player
        self.legal_moves: Set[Tuple[int, int]] = set()
        self.__update_legal_moves()
        self.game_over: bool = False
        self.num_pieces: int = int(self.state.sum())

    def __detect_game_over(self) -> None:
        # Check full board
        if self.num_pieces == 64:
            self.game_over = True
            return
        # Check if current player has legal moves
        if self.legal_moves:
            return
        # Check if opponent has legal moves
        temp_potential_moves = self.potential_moves.copy()
        temp_legal_moves = self.legal_moves.copy()
        temp_state = self.state.copy()
        self.state[0] = temp_state[1]
        self.state[1] = temp_state[0]
        self.__update_potential_moves()
        self.__update_legal_moves()
        if not self.legal_moves:
            self.game_over = True
            return
        # Restore the original game state
        self.potential_moves = temp_potential_moves
        self.legal_moves = temp_legal_moves
        self.state = temp_state

    def __update_player(self) -> None:
        # Swaps the order of the 2 grids in self.state, since the current player is always 0
        temp_state = self.state.copy()
        self.state[0] = temp_state[1]
        self.state[1] = temp_state[0]

    def __check_captures(self, opponent: int, move: Tuple[int, int]) -> bool:
        # Assumes that opponent is either 0 or 1
        # Assumes that move is a part of the potential moves for the current player (non-opponent)
        # Check if the move captures any opponent's pieces
        for di, dj in Board.DIRECTIONS:
            ni, nj = move[0] + di, move[1] + dj
            found_opponent = False
            while 0 <= ni < 8 and 0 <= nj < 8:
                if self.state[opponent, ni, nj] > 0:
                    found_opponent = True
                    ni += di
                    nj += dj
                    continue
                if self.state[1 - opponent, ni, nj] > 0:
                    if found_opponent:
                        return True
                    break
                break
        return False

    def __check_legal_move(self, move: Tuple[int, int]) -> bool:
        # Assumes that the move is in bounds
        # Check if the move is in the set of potential moves (also ensures that the square is empty)
        if move not in self.potential_moves:
            return False
        if not self.__check_captures(1, move):
            return False
        return True

    def make_move(self, move: Tuple[int, int]) -> None:
        # Assumes that the move is legal
        self.state[0, move[0], move[1]] = np.float32(1.0)

        # Flip the opponent's pieces
        for di, dj in Board.DIRECTIONS:
            ni, nj = move[0] + di, move[1] + dj
            found_opponent: bool = False
            while 0 <= ni < 8 and 0 <= nj < 8:
                if self.state[1, ni, nj] > 0:
                    found_opponent = True
                    ni += di
                    nj += dj
                    continue
                if self.state[0, ni, nj] > 0 and found_opponent:
                    flip_i, flip_j = move[0] + di, move[1] + dj
                    while (flip_i != ni or flip_j != nj):
                        self.state[1, flip_i, flip_j] = np.float32(0.0)
                        flip_i += di
                        flip_j += dj
                    break
                break

        # Update game state information
        self.num_pieces += 1
        self.empty_squares.remove(move)
        self.__update_player()
        self.__update_potential_moves()
        self.__update_legal_moves()
        self.__detect_game_over()

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
    
    def __initialize_empty_squares(self) -> None:
        # Initialize the set of empty squares based on the current state
        for i in range(8):
            for j in range(8):
                if np.isclose(self.state[0, i, j], 0.0, rtol=1e-09, atol=1e-09) and \
                   np.isclose(self.state[1, i, j], 0.0, rtol=1e-09, atol=1e-09):
                    self.empty_squares.add((i, j))
    
    def __update_potential_moves(self) -> None:
        # Assumes that self.state is up to date
        # Update the set of potential moves for the current player
        # Potential moves are empty squares that neighbor an opponent's piece
        for i, j in self.empty_squares:
            for di, dj in Board.DIRECTIONS:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= 8 or nj < 0 or nj >= 8:
                    continue
                if self.state[1, ni, nj] > 0:
                    self.potential_moves.add((i, j))

    def __update_legal_moves(self) -> None:
        # Assumes that self.potential_moves has been updated
        # Check which of the potential moves are actually legal
        for i, j in self.potential_moves:
            if self.__check_legal_move((i, j)):
                self.legal_moves.add((i, j))