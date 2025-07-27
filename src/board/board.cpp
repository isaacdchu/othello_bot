#include "board.h"
#define ONE uint64_t(1) // Guarantees that ONE is a 64-bit unsigned integer

Board::Board(State initial_state) {
    state = initial_state;
    current_player = true; // Start with black
    update_legal_moves();
    game_over = false; // This should be determined based on the initial state
}

void Board::pretty_print() const {
    std::string all_squares = "";
    for (int i = 0; i < 64; ++i) {
        if (state.black & (ONE << i)) {
            all_squares += "B ";
        } else if (state.white & (ONE << i)) {
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

void Board::make_move(uint64_t move) {
    // Implementation for making a move
    // Assumes that the given move is valid
    // Updates state, current_player, legal_moves, and game_over accordingly
    uint64_t &player = current_player ? state.black : state.white;
    player |= move;
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
    // Updates legal moves for the current player based on the current state
    uint64_t empty_squares = ~(state.black | state.white); // 1 in empty squares, 0 in occupied squares
    uint64_t &player = current_player ? state.black : state.white;
    uint64_t &opponent = current_player ? state.white : state.black;
    legal_moves = legal_moves_edges(empty_squares, player, opponent) | legal_moves_diagonals(empty_squares, player, opponent);
}

uint64_t Board::legal_moves_edges(const uint64_t& empty_squares, const uint64_t& player, const uint64_t& opponent) const {
    uint64_t legal_moves_edges = 0;
    // Moves where piece is placed north of an opponent's piece
    uint64_t legal_moves_n = 0;
    uint64_t mask_n = 0x00FFFFFFFFFFFF00;
    // Get empty squares north of opponent's pieces
    uint64_t potential_n = ((opponent & mask_n) >> 8) & empty_squares;
    int shift = 16;
    // Check that these moves capture at least one opponent's piece
    while (potential_n != 0 && shift <= 56) {
        // Check if the capture mask intersects with current player's pieces
        uint64_t capture_mask = (potential_n << shift);
        // If the capture mask intersects with player's pieces, it's a legal move
        legal_moves_n |= ((capture_mask & player) >> shift);
        // Remove these legal moves from potential_n
        potential_n &= (~legal_moves_n);
        // If we hit an empty square, remove it from potential_n
        potential_n &= (~(capture_mask | empty_squares) >> shift);
        shift += 8;
    }

    // Moves where piece is placed south of an opponent's piece
    uint64_t legal_moves_s = 0;
    uint64_t mask_s = 0x00FFFFFFFFFFFF00;
    // Get empty squares south of opponent's pieces
    uint64_t potential_s = ((opponent & mask_s) << 8) & empty_squares;
    shift = 16;
    // Check that these moves capture at least one opponent's piece
    while (potential_s != 0 && shift <= 56) {
        // Check if the capture mask intersects with current player's pieces
        uint64_t capture_mask = (potential_s >> shift);
        // If the capture mask intersects with player's pieces, it's a legal move
        legal_moves_s |= ((capture_mask & player) << shift);
        // Remove these legal moves from potential_s
        potential_s &= (~legal_moves_s);
        // If we hit an empty square, remove it from potential_s
        potential_s &= (~(capture_mask | empty_squares) << shift);
        shift += 8;
    }

    // Moves where piece is placed east of an opponent's piece
    uint64_t legal_moves_e = 0;
    uint64_t mask_e_1 = 0x7E7E7E7E7E7E7E7E;
    uint64_t mask_e_2 = 0x3F3F3F3F3F3F3F3F;
    // Get empty squares east of opponent's pieces
    uint64_t potential_e = ((opponent & mask_e_1) << 1) & empty_squares;
    shift = 2;
    // Check that these moves capture at least one opponent's piece
    while (potential_e != 0 && shift <= 7) {
        // Check if the capture mask intersects with current player's pieces
        uint64_t capture_mask = ((potential_e >> shift) & mask_e_2);
        // If the capture mask intersects with player's pieces, it's a legal move
        legal_moves_e |= (capture_mask & player) << shift;
        // Remove these legal moves from potential_e
        potential_e &= (~legal_moves_e);
        // If we hit an empty square, remove it from potential_e
        potential_e &= (~(capture_mask | empty_squares) << shift);
        shift += 1;
    }

    // Moves where piece is placed west of an opponent's piece
    uint64_t legal_moves_w = 0;
    uint64_t mask_w_1 = 0x7E7E7E7E7E7E7E7E;
    uint64_t mask_w_2 = 0xFCFCFCFCFCFCFCFC;
    // Get empty squares west of opponent's pieces
    uint64_t potential_w = ((opponent & mask_w_1) >> 1) & empty_squares;
    shift = 2;
    // Check that these moves capture at least one opponent's piece
    while (potential_w != 0 && shift <= 7) {
        // Check if the capture mask intersects with current player's pieces
        uint64_t capture_mask = ((potential_w << shift) & mask_w_2);
        // If the capture mask intersects with player's pieces, it's a legal move
        legal_moves_w |= (capture_mask & player) >> shift;
        // Remove these legal moves from potential_w
        potential_w &= (~legal_moves_w);
        // If we hit an empty square, remove it from potential_w
        potential_w &= (~(capture_mask | empty_squares) >> shift);
        shift += 1;
    }

    legal_moves_edges = legal_moves_n | legal_moves_s | legal_moves_e | legal_moves_w;
    return legal_moves_edges;
    // return legal_moves_n;
}

uint64_t Board::legal_moves_diagonals(const uint64_t& empty_squares, const uint64_t& player, const uint64_t& opponent) const {
    uint64_t legal_moves_d = 0;
    // Implementation for calculating legal moves on diagonals
    return legal_moves_d;
}