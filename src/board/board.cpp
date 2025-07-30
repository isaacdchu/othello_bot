#include "board.h"
#define ONE uint64_t(1) // Guarantees that ONE is a 64-bit unsigned integer
#define MASK_TOP_ROW uint64_t(0x00000000000000FFULL) // Top row has 1's, rest are 0's
#define MASK_BOTTOM_ROW uint64_t(0xFF00000000000000ULL) // Bottom row has 1's, rest are 0's
#define MASK_LEFT_COLUMN uint64_t(0x0101010101010101ULL) // Left column has 1's, rest are 0's
#define MASK_RIGHT_COLUMN uint64_t(0x8080808080808080ULL) // Right column has 1's, rest are 0's

Board::Board(State initial_state, bool current_player) {
    state = initial_state;
    this->current_player = current_player;
    update_legal_moves();
    uint64_t temp_legal_moves = legal_moves;
    game_over = false;
    detect_game_over();
    this->current_player = current_player; // Restore the current player after detecting game over
    this->legal_moves = temp_legal_moves; // Restore legal moves after game over detection
    // In theory, legal_moves should not be 0 here unlesss the game is already over
    // This logic depends on valid initial_state, current_player, update_legal_moves, and detect_game_over methods
}

void Board::pretty_print() const {
    std::string all_squares = "";
    for (int i = 0; i < 64; i++) {
        if (i % 8 == 0) {
            all_squares = all_squares + std::to_string((i / 8) + 1) + " ";
        }
        if (state.black & (ONE << i)) {
            all_squares += "B ";
        } else if (state.white & (ONE << i)) {
            all_squares += "W ";
        } else {
            all_squares += ". ";
        }
        if (i % 8 == 7 && i != 63) {
            all_squares += "\n";
        }
    }
    const std::string current_player_str = current_player ? "Black" : "White";
    std::cout << "Current player: " << current_player_str << std::endl;
    std::cout << "Board state:" << std::endl;
    std::cout << "  a b c d e f g h" << std::endl;
    std::cout << all_squares << std::endl;
    
}

Board Board::deep_copy() const {
    Board copy(state, current_player);
    return copy;
}

void Board::make_move(uint64_t move) {
    // Assumes that the given move is valid
    // Updates state, current_player, legal_moves, and game_over accordingly
    uint64_t &player = current_player ? state.black : state.white;
    uint64_t &opponent = current_player ? state.white : state.black;
    player |= move; // Place the piece for the current player
    legal_moves &= ~move; // Remove the move from legal moves
    uint64_t empty_squares = ~(state.black | state.white);
    flip_pieces(move, player, opponent, empty_squares);
    // Check for full board
    if (~(state.black + state.white) == 0) {
        game_over = true; // No empty squares left, game is over
    }
    detect_game_over(); // Update game state
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

const std::pair<const int, const int> Board::get_scores() const {
    // Returns the scores of both players, [black_score, white_score]
    const int black_count = __builtin_popcountll(state.black);
    const int white_count = __builtin_popcountll(state.white);
    return {black_count, white_count};
}

void Board::update_legal_moves() {
    const uint64_t empty_squares = ~(state.black | state.white);
    const uint64_t &player = current_player ? state.black : state.white;
    const uint64_t &opponent = current_player ? state.white : state.black;
    
    static const uint64_t top_two_rows = 0xFFFFFFFFFFFF0000ULL;
    static const uint64_t bottom_two_rows = 0x0000FFFFFFFFFFFFULL;
    static const uint64_t left_two_columns = 0xFCFCFCFCFCFCFCFCULL;
    static const uint64_t right_two_columns = 0x3F3F3F3F3F3F3F3FULL;
    static uint64_t temp;
    static uint64_t moves = 0;
    static auto generate_moves_right_shift = [&](const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
        temp = (((player & initial_mask) >> shift) & opponent) >> shift;
        for (int i = 0; i < 6; i++) {
            temp &= wrap_mask; // Prevent wrapping
            moves |= temp;
            temp = (temp & opponent) >> shift; // Shift to check next row
        }
    };
    static auto generate_moves_left_shift = [&](const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
        temp = (((player & initial_mask) << shift) & opponent) << shift;
        for (int i = 0; i < 6; i++) {
            temp &= wrap_mask; // Ignore left two columns to prevent wrapping
            moves |= temp;
            temp = (temp & opponent) << shift; // Shift to check next row
        }
    };
    // Legal moves that capture pieces below itself
    generate_moves_right_shift(top_two_rows, bottom_two_rows, 8);
    // Legal moves that capture pieces above itself
    generate_moves_left_shift(bottom_two_rows, top_two_rows, 8);
    // Legal moves that capture pieces to the right of itself
    generate_moves_right_shift(left_two_columns, right_two_columns, 1);
    // Legal moves that capture pieces to the left of itself
    generate_moves_left_shift(right_two_columns, left_two_columns, 1);

    // Legal moves that capture pieces diagonally down-right
    generate_moves_right_shift(top_two_rows | left_two_columns, bottom_two_rows | right_two_columns, 9);
    // Legal moves that capture pieces diagonally down-left
    generate_moves_right_shift(top_two_rows | right_two_columns, bottom_two_rows | left_two_columns, 7);
    // Legal moves that capture pieces diagonally up-right
    generate_moves_left_shift(bottom_two_rows | left_two_columns, top_two_rows | right_two_columns, 7);
    // Legal moves that capture pieces diagonally up-left
    generate_moves_left_shift(bottom_two_rows | right_two_columns, top_two_rows | left_two_columns, 9);

    legal_moves = moves & empty_squares; // Only consider moves on empty squares
}

    void Board::detect_game_over() {
    // Updates game_over and current_player based on the current state
    const uint64_t current_legal_moves = legal_moves;
    current_player = !current_player; // Switch player
    update_legal_moves(); // Update legal moves for the new current player
    if (legal_moves > 0) {
        return; // If there are legal moves, the game is not over
    }
    current_player = !current_player; // Switch back if no legal moves for the other player
    if (current_legal_moves > 0) {
        legal_moves = current_legal_moves; // Restore legal moves
        return; // Enemy passes their turn, so the game is not over
    }
    game_over = true; // No legal moves for both players, game is over (legal_moves is still 0, no restoration needed)
}

void Board::flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent, const uint64_t& empty_squares) {
    // This function flips the opponent's pieces in all directions based on the move
    // Assumes that the move is valid and already occupied by "player"
    uint64_t flips = 0;
    flips = flips & opponent; // Only flips pieces that are currently occupied by the opponent
    player |= flips;
    opponent &= ~flips;
}