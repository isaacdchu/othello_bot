
from typing import Dict, List, Tuple
from src.board import Board
from src.player import Player
from src.otto import Otto
from src.mcts import MCTSNode

def main() -> None:
    board: Board = Board(size=8)
    player_1: Player = Otto(name="black", player_id=0)
    player_2: Player = Otto(name="white", player_id=1)
    players: Dict[int, Player] = {0: player_1, 1: player_2}
    current_player: int = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")