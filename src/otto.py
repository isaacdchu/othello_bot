from typing import Tuple
import numpy as np
from multiprocessing import Pool
from src.player import Player
from src.board import Board
from src.mcts import MCTSNode

# This function simulates a number of games from the current board state
def build_tree(args) -> MCTSNode:
    board, player_id, simulations_per_worker, c = args
    local_root = MCTSNode(board, root_player=player_id)
    for _ in range(simulations_per_worker):
        node: MCTSNode = local_root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child(c=c)
        if not node.is_terminal() and not node.is_fully_expanded():
            node.expand()
            move: Tuple[int, int] = list(node.children.keys())[-1]
            node = node.children[move]
        node.rollout()
    return local_root

class Otto(Player):
    def __init__(self, name: str, player_id: int, c: float | np.float32 = 1.4, num_workers: int = 1) -> None:
        super().__init__(name, player_id)
        self.c: np.float32 = np.float32(c)
        self.num_workers: int = num_workers

    def get_move(self, board: Board) -> Tuple[int, int]:
        if len(board.legal_moves) == 1:
            return next(iter(board.legal_moves))
        
        root = MCTSNode(board, root_player=self.player_id)
        num_simulations = self.__calculate_simulation_count(board.num_pieces)
        simulations_per_worker = max(num_simulations // self.num_workers, 1)

        # Use multiprocessing to parallelize the simulation
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(build_tree, [(board, self.player_id, simulations_per_worker, self.c)] * self.num_workers)

        # Merge the results from all workers into the root node
        for local_root in results:
            for move, child in local_root.children.items():
                if move not in root.children:
                    root.children[move] = child
                else:
                    root.children[move].visit_count += child.visit_count
                    root.children[move].value_sum += child.value_sum

        # Choose the move with the highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move
    
    def __calculate_simulation_count(self, num_pieces: int) -> int:
        """Returns the number of simulations performed, adapting to game phase."""
        if num_pieces == 4:
            return 1 # All first moves are equal
        # Opening: very few pieces, less simulations
        if num_pieces <= 10:
            return 500
        # Endgame: almost full board, less simulations
        elif num_pieces >= 54:
            return 750
        # Midgame: most simulations
        else:
            # Perform 1000 simulations for midgame
            return 1000