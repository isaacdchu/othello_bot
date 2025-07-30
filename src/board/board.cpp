#include "board.h"

// Constants/Masks filled with 1s except for the name
static const uint64_t top_two_rows = 0xFFFFFFFFFFFF0000ULL;
static const uint64_t bottom_two_rows = 0x0000FFFFFFFFFFFFULL;
static const uint64_t right_two_columns = 0x3F3F3F3F3F3F3F3FULL;
static const uint64_t left_two_columns = 0xFCFCFCFCFCFCFCFCULL;

static const uint64_t top_row = 0xFFFFFFFFFFFFFF00ULL;
static const uint64_t bottom_row = 0x00FFFFFFFFFFFFFFFULL;
static const uint64_t right_column = 0x7F7F7F7F7F7F7F7FULL;
static const uint64_t left_column = 0xFEFEFEFEFEFEFEFEULL;

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
        if (state.black & (uint64_t(1) << i)) {
            all_squares += "B ";
        } else if (state.white & (uint64_t(1) << i)) {
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
    flip_pieces(move, player, opponent);
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
    legal_moves = 0; // Reset legal moves at the start
    const uint64_t empty_squares = ~(state.black | state.white);
    const uint64_t &player = current_player ? state.black : state.white;
    const uint64_t &opponent = current_player ? state.white : state.black;
    
    // Legal moves that capture pieces below itself
    generate_moves_right_shift(player, opponent, top_two_rows, bottom_two_rows, 8);
    // Legal moves that capture pieces above itself
    generate_moves_left_shift(player, opponent, bottom_two_rows, top_two_rows, 8);
    // Legal moves that capture pieces to the right of itself
    generate_moves_right_shift(player, opponent, left_two_columns, right_two_columns, 1);
    // Legal moves that capture pieces to the left of itself
    generate_moves_left_shift(player, opponent, right_two_columns, left_two_columns, 1);

    // Legal moves that capture pieces diagonally down-right
    generate_moves_right_shift(player, opponent, top_two_rows | left_two_columns, bottom_two_rows | right_two_columns, 9);
    // Legal moves that capture pieces diagonally down-left
    generate_moves_right_shift(player, opponent, top_two_rows | right_two_columns, bottom_two_rows | left_two_columns, 7);
    // Legal moves that capture pieces diagonally up-right
    generate_moves_left_shift(player, opponent, bottom_two_rows | left_two_columns, top_two_rows | right_two_columns, 7);
    // Legal moves that capture pieces diagonally up-left
    generate_moves_left_shift(player, opponent, bottom_two_rows | right_two_columns, top_two_rows | left_two_columns, 9);
    legal_moves &= empty_squares; // Ensure legal moves are only on empty squares
}

void Board::generate_moves_right_shift(const uint64_t& player, const uint64_t& opponent, 
    const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
    // Should only be called by update_legal_moves, nowhere else
    uint64_t temp_moves = 0;
    uint64_t temp = (((player & initial_mask) >> shift) & opponent) >> shift;
    for (int i = 0; i < 6; i++) {
        temp &= wrap_mask; // Prevent wrapping
        temp_moves |= temp;
        temp = (temp & opponent) >> shift; // Shift to check next row
    }
    legal_moves |= temp_moves; // Combine all legal moves found
}

void Board::generate_moves_left_shift(const uint64_t& player, const uint64_t& opponent, 
    const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
    // Should only be called by update_legal_moves, nowhere else
    uint64_t temp_moves = 0;
    uint64_t temp = (((player & initial_mask) << shift) & opponent) << shift;
    for (int i = 0; i < 6; i++) {
        temp &= wrap_mask; // Prevent wrapping
        temp_moves |= temp;
        temp = (temp & opponent) << shift; // Shift to check next row
    }
    legal_moves |= temp_moves; // Combine all legal moves found
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

void Board::flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent) {
    // This function flips the opponent's pieces in all directions based on the move
    // Assumes that the move is valid and already occupied by "player"
    flip_pieces_right_shift(move, player, opponent, top_two_rows, bottom_row, 8); // Flips pieces above the move
    flip_pieces_left_shift(move, player, opponent, bottom_two_rows, top_row, 8); // Flips pieces below the move
    flip_pieces_right_shift(move, player, opponent, right_two_columns, left_column, 1); // Flips pieces to the right of the move
    flip_pieces_left_shift(move, player, opponent, left_two_columns, right_column, 1); // Flips pieces to the left of the move

    // Flips pieces diagonally down-right
    flip_pieces_left_shift(move, player, opponent, bottom_two_rows | right_two_columns, top_row | left_column, 9);
    // Flips pieces diagonally down-left
    flip_pieces_left_shift(move, player, opponent, bottom_two_rows | left_two_columns, top_row | right_column, 7);
    // Flips pieces diagonally up-right
    flip_pieces_right_shift(move, player, opponent, top_two_rows | right_two_columns, bottom_row | left_column, 7);
    // Flips pieces diagonally up-left
    flip_pieces_right_shift(move, player, opponent, top_two_rows | left_two_columns, bottom_row | right_column, 9);
}

void Board::flip_pieces_right_shift(const uint64_t& move, uint64_t& player, uint64_t& opponent, 
    const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
    // Should only be called by flip_pieces, nowhere else
    uint64_t flips = 0;
    // Stores current square to check for flips
    uint64_t temp = ((move & initial_mask) >> shift) & opponent;
    while (temp > 0) {
        flips |= temp;
        temp = ((temp & wrap_mask) >> shift);
        if (temp & player) {
            // We hit our own pieces, so we can flip the opponent's pieces
            player |= flips;
            opponent &= ~flips;
            break;
        }
        temp &= opponent; // Continue only with opponent's pieces
    }
}

void Board::flip_pieces_left_shift(const uint64_t& move, uint64_t& player, uint64_t& opponent, 
    const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift) {
    // Should only be called by flip_pieces, nowhere else
    uint64_t flips = 0;
    // Stores current square to check for flips
    uint64_t temp = ((move & initial_mask) << shift) & opponent;
    while (temp > 0) {
        flips |= temp;
        temp = ((temp & wrap_mask) << shift);
        if (temp & player) {
            // We hit our own pieces, so we can flip the opponent's pieces
            player |= flips;
            opponent &= ~flips;
            break;
        }
        temp &= opponent; // Continue only with opponent's pieces
    }
}