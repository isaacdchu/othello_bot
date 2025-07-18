from typing import Dict

from src.board import Board
from src.player import Player
from src.human import Human

def main() -> None:
    board: Board = Board()
    player_1: Player = Human("Player 1")
    player_2: Player = Human("Player 2")
    players: Dict[int, Player] = {0: player_1, 1: player_2}
    current_player: Player = player_1
    while not board.game_over:
        board.pretty_print()
        move = current_player.get_move(board)
        if move in board.legal_moves:
            board.make_move(move)
            current_player = players[board.current_player]
        else:
            print("Invalid move. Please try again.")
    print("Game over!")
    print("Final board state:")
    board.pretty_print()
    print(f"Scores: {player_1}: {board.state[0].sum()}, {player_2}: {board.state[1].sum()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")