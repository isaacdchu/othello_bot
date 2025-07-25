from typing import Dict, List
from time import perf_counter
from src.board import Board
from src.player import Player
from src.human import Human
from src.otto import Otto
from src.randobot import Randobot

def main() -> None:
    board: Board = Board()
    # player_1: Player = Human("Player 1", player_id=0)
    player_1: Player = Otto("Player 1", player_id=0, c=1.2, num_workers=10)
    # player_2: Player = Otto("Player 2", player_id=1, c=1.2, num_workers=10)
    player_2: Player = Randobot("Player 2", player_id=1)
    players: Dict[int, Player] = {0: player_1, 1: player_2}
    colors: Dict[int, str] = {0: "Black", 1: "White"}
    current_player: Player = player_1
    times: List[float] = []
    while not board.game_over:
        print(f"{colors[board.current_player]} to move")
        board.pretty_print()
        print()
        start_time = perf_counter()
        move = current_player.get_move(board)
        end_time = perf_counter()
        print(f"{current_player.name} chose move ({move[0]}, {move[1]}) in {end_time - start_time:.2f} seconds")
        times.append(end_time - start_time)
        if move in board.legal_moves:
            board.make_move(move)
            current_player = players[board.current_player]
        else:
            print("Invalid move. Please try again.")
    print("Game over!")
    print("Final board state:")
    board.pretty_print()
    winner = board.get_winner()
    if winner is not None:
        print(f"{players[winner].name} wins!")
    else:
        print("It's a draw!")
    print("Final scores:")
    scores = board.get_scores()
    print(f"Player 1 (Black): {scores[0]}")
    print(f"Player 2 (White): {scores[1]}")
    print(times)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")