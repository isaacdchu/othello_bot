from src.board import Board

def main() -> None:
    board = Board()
    board.pretty_print(coords=True)
    print("Potential moves for black:", board.potential_moves)
    print("Legal moves for black:", board.legal_moves)

if __name__ == "__main__":
    main()