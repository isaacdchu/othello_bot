from __future__ import annotations
from typing import List, Set, Tuple, Optional
import numpy as np
from src.config_handler import get_config

class Board:
    DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    def __init__(self, size: int = 8) -> None:
        self.size: int = size
        # Black's pieces are represented by self.state[0] to begin
        self.state: np.ndarray = Board.__convert_to_state(get_config("config.yaml").board.starting_position)
        # Stores the set of all empty squares
        self.empty_squares: Set[Tuple[int, int]] = set()
        self.__initialize_empty_squares()
        # Stores the set of legal moves for the current player
        self.legal_moves: Set[Tuple[int, int]] = set()
        self.__update_legal_moves()
        self.game_over: bool = False
        self.num_pieces: int = int(self.state.sum())
        self.current_player: int = 0 # 0 for black, 1 for white
    
    def __hash__(self) -> int:
        """Hash function for the board state."""
        return hash(self.state.tobytes())
    
    def __eq__(self, other: object) -> bool:
        """Equality check for the board state."""
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.state, other.state) and \
               self.current_player == other.current_player

    def copy(self) -> Board:
        """Create a deep copy of the board."""
        new_board = Board(size=self.size)
        new_board.state = self.state.copy()
        new_board.empty_squares = self.empty_squares.copy()
        new_board.legal_moves = self.legal_moves.copy()
        new_board.game_over = self.game_over
        new_board.num_pieces = self.num_pieces
        new_board.current_player = self.current_player
        return new_board

    def __opponent_has_legal_moves(self) -> bool:
        # Check if the opponent has any legal moves
        for move in self.empty_squares:
            for di, dj in Board.DIRECTIONS:
                ni, nj = move[0] + di, move[1] + dj
                found_current_player = False
                while 0 <= ni < 8 and 0 <= nj < 8:
                    if self.state[0, ni, nj] > 0:
                        found_current_player = True
                        ni += di
                        nj += dj
                        continue
                    if self.state[1, ni, nj] > 0:
                        if found_current_player:
                            return True
                        break
                    break
        return False

    def __detect_game_over(self) -> None:
        # Check full board
        if self.num_pieces == 64:
            self.game_over = True
            return
        # Check if current player has legal moves
        if self.legal_moves:
            return
        # Check if opponent has legal moves
        if self.__opponent_has_legal_moves():
            # If current player has no legal moves but opponent does, switch players
            self.__update_player()
            self.__update_legal_moves()
            return
        # If neither player has legal moves, the game is over
        self.game_over = True

    def __update_player(self) -> None:
        # Swaps the order of the 2 grids in self.state, since the current player is always 0
        temp_state = self.state.copy()
        self.state[0] = temp_state[1]
        self.state[1] = temp_state[0]
        self.current_player = 1 - self.current_player

    def __check_legal_move(self, move: Tuple[int, int]) -> bool:
        # Assumes that move is a potential move for the current player
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
    
    def get_winner(self) -> Optional[int]:
        """Returns the winner of the game (0 for black, 1 for white), or None if it's a draw."""
        if not self.game_over:
            return None
        black_score, white_score = self.get_scores()
        if black_score > white_score:
            return 0
        elif white_score > black_score:
            return 1
        else:
            return None

    def evaluate(self) -> np.float32:
        """Evaluates the board state for the current player."""
        return np.float32((self.state[0] - self.state[1]).sum() / 64.0)

    def make_move(self, move: Tuple[int, int]) -> None:
        # Assumes that the move is legal
        # Places a new piece on the board
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
                        self.state[0, flip_i, flip_j] = np.float32(1.0)
                        self.state[1, flip_i, flip_j] = np.float32(0.0)
                        flip_i += di
                        flip_j += dj
                    break
                break

        # Update game state information
        self.num_pieces += 1
        self.empty_squares.remove(move)
        self.__update_player()
        self.__update_legal_moves()
        self.__detect_game_over()
    
    def get_scores(self) -> Tuple[int, int]:
        # Returns the scores of both players in (black_score, white_score) format
        if self.current_player == 0:
            black_score = int(self.state[0].sum())
            white_score = int(self.state[1].sum())
        else:
            white_score = int(self.state[0].sum())
            black_score = int(self.state[1].sum())
        return black_score, white_score

    def pretty_print(self, coords: bool = True) -> None:
        board_repr: List[List[str]] = []
        for i in range(8):
            row: List[str] = []
            for j in range(8):
                if self.state[self.current_player, i, j] == 1.0:
                    row.append('B')
                elif self.state[1 - self.current_player, i, j] == 1.0:
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
        # Converts a board representation to the state format used by the Board class
        black: np.ndarray = np.zeros((8, 8), dtype=np.float32)
        white: np.ndarray = np.zeros((8, 8), dtype=np.float32)
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == 'B':
                    black[i, j] = np.float32(1.0)
                elif cell == 'W':
                    white[i, j] = np.float32(1.0)
        state: np.ndarray = np.array([black, white], dtype=np.float32)
        return state
    
    def __initialize_empty_squares(self) -> None:
        occupied: np.ndarray = self.state[0] + self.state[1]  # sum over channels
        empty_mask: np.ndarray = occupied == 0.0
        self.empty_squares = set(zip(*np.where(empty_mask)))

    def __update_legal_moves(self) -> None:
        # Assumes that self.state is up to date
        # Assumes that self.empty_squares is up to date
        self.legal_moves.clear()
        for move in self.empty_squares:
            if self.__check_legal_move(move):
                self.legal_moves.add(move)