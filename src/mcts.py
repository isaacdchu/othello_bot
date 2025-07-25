
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import concurrent.futures
from os import cpu_count
import numpy as np
import random
from src.board import Board

def single_rollout(args) -> np.float32:
    board, root_player, depth = args
    board = board.copy()
    for _ in range(depth):
        if board.game_over:
            break
        legal_moves: List[Tuple[int, int]] = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.make_move(move)
    return heuristic_value(board, board.get_winner(), root_player)

def heuristic_value(board: Board, winner: Optional[int], root_player: int) -> np.float32:
        """Returns the heuristic value of the board state for the root player."""
        # Root player's pieces - opponent's pieces
        piece_difference: np.float32 = board.evaluate() * np.float32(1.0 if board.current_player == root_player else -1.0)
        # Game progress weight (early vs late game). Score matters more in the late game.
        score_weight: np.float32 = np.float32(board.num_pieces / 64.0) # ranges from 0.0 to 1.0
        # +1 if the root player is the winner, -1 if the opponent is the winner, 0 if it's a draw
        outcome_value: np.float32 = np.float32(0.0) # ranges from -1.0 to 1.0
        if winner is not None:
            outcome_value = np.float32(1.0 if winner == root_player else -1.0)
        return np.float32(piece_difference * score_weight + outcome_value * (1.0 - score_weight))

class MCTSNode:
    def __init__(self, 
                 board: Board, 
                 root_player: int, 
                 parent: Optional[MCTSNode] = None, 
                 ) -> None:
        self.board: Board = board # The board state at this node, should not be modified
        self.root_player: int = root_player # The player who we want the best move for (0 for black, 1 for white)
        self.parent: Optional[MCTSNode] = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.untried_moves: Set[Tuple[int, int]] = board.legal_moves.copy()
        self.visit_count: int = 0
        self.value_sum: np.float32 = np.float32(0.0)

    def select_child(self, c: np.float32 = np.float32(1.4)) -> MCTSNode:
        children: List[MCTSNode] = list(self.children.values())
        visit_counts = np.array([child.visit_count for child in children], dtype=np.float32)
        # If there are unvisited children, select one randomly
        if any(visit_counts == 0):
            unvisited_children = [child for child in children if child.visit_count == 0]
            return random.choice(unvisited_children)
        # Calculate UCT values for each child
        value_sums = np.array([child.value_sum for child in children], dtype=np.float32)
        total_visits = np.sum(visit_counts)
        exploration = np.sqrt(np.log(total_visits)) / (visit_counts)
        uct_values: np.ndarray = value_sums / (visit_counts) + c * exploration
        return children[np.argmax(uct_values)]
    
    def expand(self) -> None:
        # Expands the node by adding exactly one child
        if not self.untried_moves:
            return
        move = self.untried_moves.pop()
        new_board: Board = self.board.copy()
        new_board.make_move(move)
        self.children[move] = MCTSNode(new_board, self.root_player, self)

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.board.legal_moves)

    def rollout(self, depth: int = 64) -> None:
        board = self.board.copy()
        for _ in range(depth):
            if board.game_over:
                break
            legal_moves = list(board.legal_moves)
            # Policy: choose a random legal move
            move = random.choice(legal_moves)
            board.make_move(move)
        self.backpropagate(self.heuristic_value(board.get_winner()))

    def parallel_rollout(self, num_rollouts: int, depth: int = 64) -> None:
        args = [(self.board.copy(), self.root_player, depth) for _ in range(num_rollouts)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(single_rollout, args))
        for value in results:
            self.backpropagate(value)

    def backpropagate(self, value: np.float32) -> None:
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def is_terminal(self) -> bool:
        return self.board.game_over
    
    def heuristic_value(self, winner: Optional[int]) -> np.float32:
        """Returns the heuristic value of the board state for the root player."""
        # Root player's pieces - opponent's pieces
        piece_difference: np.float32 = self.board.evaluate() * np.float32(1.0 if self.board.current_player == self.root_player else -1.0)
        # Game progress weight (early vs late game). Score matters more in the late game.
        score_weight: np.float32 = np.float32(self.board.num_pieces / 64.0) # ranges from 0.0 to 1.0
        # +1 if the root player is the winner, -1 if the opponent is the winner, 0 if it's a draw
        outcome_value: np.float32 = np.float32(0.0) # ranges from -1.0 to 1.0
        if winner is not None:
            outcome_value = np.float32(1.0 if winner == self.root_player else -1.0)
        return np.float32(piece_difference * score_weight + outcome_value * (1.0 - score_weight))
