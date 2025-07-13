from __future__ import annotations
from typing import Dict, Tuple
from src.othello import Board

def parse_move(move: str) -> Tuple[int, int]:
    letter, number = move[0], move[1]
    col = ord(letter) - ord('a')
    row = 8 - int(number)
    return (row, col)

def get_valid_move(player: str, invalid_moves: Dict[str, None]) -> Tuple[str, int, int]:
    while True:
        move = input(f"Player {player}, enter your move (e.g., 'a1'): ").lower().strip()[:2]
        if move == 'qu':
            return 'qu', -1, -1
        if move in invalid_moves:
            print("Invalid move. Try again.")
            continue
        try:
            row, col = parse_move(move)
            return move, row, col
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid move (e.g., 'a1').")

def switch_player(player: str) -> str:
    return 'W' if player == 'B' else 'B'

def main():
    board = Board()
    print("Enter 'qu' to exit the game.")
    print("Black moves first.")
    player = 'B'  # Start with Black player
    invalid_moves: Dict[str, None] = {"d4": None, "d5": None, "e4": None, "e5": None}
    while True:
        if not board.can_move('B') and not board.can_move('W'):
            print("No valid moves left for both players. Game over.")
            print(board.print_board())
            result = board.game_result()
            print(f"Game result: {result[0]} (B: {result[1]}, W: {result[2]})")
            break
        if not board.can_move(player):
            print(f"Player {player} has no valid moves. Skipping turn.")
            player = switch_player(player)
            continue
        print(board.print_board())
        move, row, col = get_valid_move(player, invalid_moves)
        if move == 'qu':
            print("Game ended.")
            break
        if not board.make_move(player, row, col):
            print("Invalid move. Try again.")
            continue
        player = switch_player(player)

if __name__ == "__main__":
    main()
