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
    std::string current_player_str = current_player ? "Black" : "White";
    std::cout << "Current player: " << current_player_str << std::endl;
    std::cout << "Board state:" << std::endl;
    std::cout << "  a b c d e f g h" << std::endl;
    std::cout << all_squares << std::endl;
    
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
    uint64_t &opponent = current_player ? state.white : state.black;
    // Flip opponent's pieces in each direction
    uint64_t empty_squares = ~(state.black | state.white);
    flip_pieces(move, player, opponent, empty_squares);
    player |= move; // Place the piece for the current player
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

void Board::update_legal_moves() {
    const uint64_t empty_squares = ~(state.black | state.white);
    uint64_t &player = current_player ? state.black : state.white;
    uint64_t &opponent = current_player ? state.white : state.black;
    uint64_t moves = 0;

    // Helper lambda for a single direction
    auto gen_moves = [&](const int shift, const uint64_t mask, const bool left_shift) {
        uint64_t temp = opponent & mask;
        uint64_t t = left_shift ? (temp & (player << shift)) : (temp & (player >> shift));
        for (int i = 0; i < 5; ++i) {
            t |= left_shift ? (temp & (t << shift)) : (temp & (t >> shift));
        }
        moves |= left_shift ? (empty_squares & (t << shift)) : (empty_squares & (t >> shift));
    };

    // Call for each direction
    gen_moves(8, 0x00FFFFFFFFFFFF00ULL, false); // North
    gen_moves(8, 0x00FFFFFFFFFFFF00ULL, true);  // South
    gen_moves(1, 0x7f7f7f7f7f7f7f7fULL, true);  // East
    gen_moves(1, 0xfefefefefefefefeULL, false); // West
    gen_moves(7, 0x7f7f7f7f7f7f7f7fULL, true);  // SE
    gen_moves(7, 0xfefefefefefefefeULL, false); // NW
    gen_moves(9, 0x7f7f7f7f7f7f7f7fULL, true);  // NE
    gen_moves(9, 0xfefefefefefefefeULL, false); // SW

    legal_moves = moves;
}

void Board::detect_game_over() {
    // Updates game_over and current_player based on the current state
    const uint64_t current_legal_moves = legal_moves;
    current_player = !current_player; // Switch player
    update_legal_moves(); // Update legal moves for the new current player
    if (legal_moves != 0) {
        return; // If there are legal moves, the game is not over
    }
    current_player = !current_player; // Switch back if no legal moves for the other player
    if (current_legal_moves != 0) {
        legal_moves = current_legal_moves; // Restore legal moves
        return; // Enemy passes their turn, so the game is not over
    }
    game_over = true; // No legal moves for both players, game is over
}

void Board::flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent, const uint64_t&) {
    static const int directions[8] = {8, -8, 1, -1, 9, 7, -9, -7};
    static const uint64_t edge_masks[8] = {
        ~MASK_TOP_ROW,    // North
        ~MASK_BOTTOM_ROW, // South
        ~MASK_RIGHT_COLUMN, // East
        ~MASK_LEFT_COLUMN,  // West
        ~MASK_TOP_ROW & ~MASK_RIGHT_COLUMN,    // NE
        ~MASK_BOTTOM_ROW & ~MASK_RIGHT_COLUMN, // SE
        ~MASK_TOP_ROW & ~MASK_LEFT_COLUMN,     // NW
        ~MASK_BOTTOM_ROW & ~MASK_LEFT_COLUMN   // SW
    };

    uint64_t flips = 0;
    for (int d = 0; d < 8; ++d) {
        int shift = directions[d];
        uint64_t mask = edge_masks[d];
        uint64_t cur = move;
        uint64_t captured = 0;

        while (true) {
            // Prevent wrap-around by masking before shifting
            if ((shift > 0 && (cur & mask) == 0) || (shift < 0 && (cur & mask) == 0)) break;
            cur = (shift > 0) ? (cur >> shift) : (cur << -shift);
            if (cur == 0) break;
            if ((cur & opponent) != 0) {
                captured |= cur;
            } else if ((cur & player) != 0) {
                flips |= captured;
                break;
            } else {
                break;
            }
        }
    }
    player |= flips;
    opponent &= ~flips;
}