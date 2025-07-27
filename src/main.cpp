#include "main.h"

int main() {
    Player* player_1 = new Human("Player 1", true);
    Player* player_2 = new Human("Player 2", false);
    State initial_state = {0x810000000, 0x1008000000};
    Board board(initial_state);
    while (!board.is_game_over()) {
        run_game(player_1, player_2, board);
    }

    delete player_1;
    delete player_2;
    return 0;
}

void run_game(Player* player_1, Player* player_2, Board& board) {
    board.pretty_print();
    uint64_t move;
    if (board.get_current_player()) {
        move = player_1->get_move(board);
    } else {
        move = player_2->get_move(board);
    }
    board.make_move(move, board.get_current_player());
}