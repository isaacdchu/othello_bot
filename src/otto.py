from typing import Tuple
from os import cpu_count
import numpy as np
from src.player import Player
from src.board import Board
from src.mcts import MCTSNode

class Otto(Player):
    def __init__(self, name: str, player_id: int, c: float | np.float32 = 1.4, num_workers: int = 1) -> None:
        super().__init__(name, player_id)
        self.c: np.float32 = np.float32(c)
        self.num_workers: int = num_workers

    def get_move(self, board: Board) -> Tuple[int, int]:
        if len(board.legal_moves) == 1:
            return next(iter(board.legal_moves))
        root = MCTSNode(board, root_player=self.player_id)
        for _ in range(self.__calculate_simulation_count(board.num_pieces)):  # Number of simulations
            node = root
            # Selection: traverse down the tree using UCT until a node with untried moves or terminal
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(c=self.c)
            # Expansion: expand one child if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node.expand()
                # After expansion, move to the newly added child
                move = list(node.children.keys())[-1]
                node = node.children[move]
            # Rollout: simulate from this node
            node.parallel_rollout(num_rollouts=self.num_workers)
        # Choose the move with the highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move
    
    def __calculate_simulation_count(self, num_pieces: int) -> int:
        """Returns the number of simulations performed, adapting to game phase."""
        if num_pieces == 4:
            return 1 # All first moves are equal
        # Opening: very few pieces, less simulations
        if num_pieces <= 10:
            return int(500 // self.num_workers)
        # Endgame: almost full board, less simulations
        elif num_pieces >= 54:
            return int(750 // self.num_workers)
        # Midgame: most simulations
        else:
            # Perform 1000 simulations for midgame
            return int(1000 // self.num_workers)