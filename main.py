from __future__ import annotations
from typing import Dict
from src.player import Player
from src.human import Human
from src.othello import Othello

def main():
    player_1: Player = Human(name='Black', color='B')
    player_2: Player = Human(name='White', color='W')
    players: Dict[str, Player] = {'B': player_1, 'W': player_2}
    game = Othello(player_1, player_2)
    print("Welcome to Othello!")
    print(game.board.print_board())

    while True:
        move = players[game.current_player].get_move()
        if not game.new_turn(move):
            continue
        if game.detect_game_end():
            print("Game over!")
            black_score, white_score = game.board.get_player_scores()
            if black_score > white_score:
                print(f"{player_1.name} wins with {black_score} points!")
            elif white_score > black_score:
                print(f"{player_2.name} wins with {white_score} points!")
            else:
                print(f"It's a draw! Both players scored {black_score} points.")
            break
        print(game.board.print_board())
        print(f"Scores - Black: {player_1.score}, White: {player_2.score}")
    
    print("Thanks for playing!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated.")
    except Exception as e:
        print(f"An error occurred: {e}")
