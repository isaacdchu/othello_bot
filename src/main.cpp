#include "main.h"

int main() {
    Player* player_1 = new Human("Player 1", true);
    Player* player_2 = new Human("Player 2", false);
    // Player* player_1 = new RandoBot("RandoBot 1", true);
    // Player* player_2 = new RandoBot("RandoBot 2", false);
    // Player* player_1 = new Otto("Otto 1", true);
    // Player* player_2 = new Otto("Otto 2", false);
    State initial_state = {0x0000000810000000, 0x0000000100800000}; // Default starting position
    // State initial_state = {0x0000000000030001, 0x0000000000040200}; // test when a turn is passed
    Board board(initial_state, true); // Start with black player
    print_legal_moves(board.get_legal_moves());
    while (!board.is_game_over()) {
        run_game(player_1, player_2, board);
    }
    std::cout << "Game over!" << std::endl;
    board.pretty_print();
    int player_1_score = board.get_scores().first;
    int player_2_score = board.get_scores().second;
    std::cout << "Final scores: " << player_1->get_name() << " (" 
              << player_1_score << ") - " 
              << player_2->get_name() << " (" << player_2_score << ")" << std::endl;
    if (player_1_score > player_2_score) {
        std::cout << player_1->get_name() << " wins!" << std::endl;
    } else if (player_1_score < player_2_score) {
        std::cout << player_2->get_name() << " wins!" << std::endl;
    } else {
        std::cout << "It's a draw!" << std::endl;
    }
    // Clean up
    delete player_1;
    delete player_2;
    return 0;
}

void run_game(Player* player_1, Player* player_2, Board& board) {
    board.pretty_print();
    // Print legal moves for the current player
    const uint64_t legal_moves = board.get_legal_moves();
    print_legal_moves(legal_moves);
    uint64_t move;
    if (board.get_current_player()) {
        std::cout << player_1->get_name() << "'s turn." << std::endl;
        move = player_1->get_move(board);
        std::cout << player_1->get_name() << " chose move: " << move_to_square(move) << std::endl;
    } else {
        std::cout << player_2->get_name() << "'s turn." << std::endl;
        move = player_2->get_move(board);
        std::cout << player_2->get_name() << " chose move: " << move_to_square(move) << std::endl;
    }
    board.make_move(move);
}