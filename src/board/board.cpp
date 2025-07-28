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
    // Updates legal moves for the current player based on the current state
    uint64_t empty_squares = ~(state.black | state.white); // 1 in empty squares, 0 in occupied squares
    uint64_t &player = current_player ? state.black : state.white;
    uint64_t &opponent = current_player ? state.white : state.black;
    uint64_t moves = 0;
    // Masks for board edges
    const uint64_t notA = 0xfefefefefefefefeULL; // Not file A (left)
    const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL; // Not file H (right)

    // North (up)
    uint64_t temp = opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t t = temp & (player >> 8);
    t |= temp & (t >> 8);
    t |= temp & (t >> 8);
    t |= temp & (t >> 8);
    t |= temp & (t >> 8);
    t |= temp & (t >> 8);
    moves |= empty_squares & (t >> 8);

    // South (down)
    t = temp & (player << 8);
    t |= temp & (t << 8);
    t |= temp & (t << 8);
    t |= temp & (t << 8);
    t |= temp & (t << 8);
    t |= temp & (t << 8);
    moves |= empty_squares & (t << 8);

    // East (right)
    temp = opponent & notH;
    t = temp & (player << 1);
    t |= temp & (t << 1);
    t |= temp & (t << 1);
    t |= temp & (t << 1);
    t |= temp & (t << 1);
    t |= temp & (t << 1);
    moves |= empty_squares & (t << 1);

    // West (left)
    temp = opponent & notA;
    t = temp & (player >> 1);
    t |= temp & (t >> 1);
    t |= temp & (t >> 1);
    t |= temp & (t >> 1);
    t |= temp & (t >> 1);
    t |= temp & (t >> 1);
    moves |= empty_squares & (t >> 1);

    // Northeast (up-right)
    temp = opponent & notH;
    t = temp & (player >> 7);
    t |= temp & (t >> 7);
    t |= temp & (t >> 7);
    t |= temp & (t >> 7);
    t |= temp & (t >> 7);
    t |= temp & (t >> 7);
    moves |= empty_squares & (t >> 7);

    // Northwest (up-left)
    temp = opponent & notA;
    t = temp & (player >> 9);
    t |= temp & (t >> 9);
    t |= temp & (t >> 9);
    t |= temp & (t >> 9);
    t |= temp & (t >> 9);
    t |= temp & (t >> 9);
    moves |= empty_squares & (t >> 9);

    // Southeast (down-right)
    temp = opponent & notH;
    t = temp & (player << 9);
    t |= temp & (t << 9);
    t |= temp & (t << 9);
    t |= temp & (t << 9);
    t |= temp & (t << 9);
    t |= temp & (t << 9);
    moves |= empty_squares & (t << 9);

    // Southwest (down-left)
    temp = opponent & notA;
    t = temp & (player << 7);
    t |= temp & (t << 7);
    t |= temp & (t << 7);
    t |= temp & (t << 7);
    t |= temp & (t << 7);
    t |= temp & (t << 7);
    moves |= empty_squares & (t << 7);
    legal_moves = moves; // Update legal moves for the current player
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

void Board::flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent, const uint64_t& empty_squares) {
    // Directional constants for your bitboard orientation
    static const int directions[8] = {8, -8, 1, -1, 9, 7, -9, -7};
    static const uint64_t masks[8] = {
        ~(MASK_TOP_ROW | MASK_BOTTOM_ROW), // N/S
        ~(MASK_TOP_ROW | MASK_BOTTOM_ROW),
        ~(MASK_LEFT_COLUMN | MASK_RIGHT_COLUMN), // E/W
        ~(MASK_LEFT_COLUMN | MASK_RIGHT_COLUMN),
        ~(MASK_TOP_ROW | MASK_LEFT_COLUMN | MASK_BOTTOM_ROW | MASK_RIGHT_COLUMN), // Diagonals
        ~(MASK_TOP_ROW | MASK_LEFT_COLUMN | MASK_BOTTOM_ROW | MASK_RIGHT_COLUMN),
        ~(MASK_TOP_ROW | MASK_LEFT_COLUMN | MASK_BOTTOM_ROW | MASK_RIGHT_COLUMN),
        ~(MASK_TOP_ROW | MASK_LEFT_COLUMN | MASK_BOTTOM_ROW | MASK_RIGHT_COLUMN)
    };

    uint64_t flip = 0;
    for (int d = 0; d < 8; ++d) {
        int shift = directions[d];
        uint64_t mask = masks[d];
        uint64_t flips = 0;
        uint64_t candidate = 0;
        uint64_t m = move;

        for (int i = 0; i < 7; ++i) {
            // Shift in the correct direction
            if (shift > 0) {
                if ((m >> shift) & mask) m = (m >> shift) & mask;
                else break;
            } else {
                if ((m << -shift) & mask) m = (m << -shift) & mask;
                else break;
            }
            if (m & opponent) {
                candidate |= m;
            } else if (m & player) {
                flips |= candidate;
                break;
            } else {
                break;
            }
        }
        flip |= flips;
    }
    player |= flip;
    opponent &= ~flip;
}