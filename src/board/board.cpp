#include "board.h"

Board::Board(State initial_state) {
    state = initial_state;
    current_player = true; // Start with black
    update_legal_moves();
    game_over = false; // This should be determined based on the initial state
}

void Board::pretty_print() const {
    uint64_t one = 1;
    std::string all_squares = "";
    for (int i = 0; i < 64; ++i) {
        if (state.black & (one << i)) {
            all_squares += "B ";
        } else if (state.white & (one << i)) {
            all_squares += "W ";
        } else {
            all_squares += ". ";
        }
        if ((i + 1) % 8 == 0) {
            all_squares += "\n";
        }
    }
    std::string current_player_str = current_player ? "Black" : "White";
    std::cout << "Current player: " << current_player_str << std::endl;
    std::cout << "Board state:\n" << all_squares << std::endl;
}

Board Board::deep_copy() const {
    Board copy;
    copy.state = this->state;
    copy.current_player = this->current_player;
    copy.legal_moves = this->legal_moves;
    copy.game_over = this->game_over;
    return copy;
}

void Board::make_move(uint64_t move, bool for_player) {
    // Implementation for making a move
    // Assumes that the given move is valid
    // Updates state, current_player, legal_moves, and game_over accordingly
}

uint64_t Board::get_legal_moves() const {
    return legal_moves;
}

bool Board::get_current_player() const {
    return current_player;
}

bool Board::is_game_over() const {
    return game_over;
}

void Board::update_legal_moves() {
    // Implementation for updating legal moves based on the current state
}