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
    uint64_t &opponent = current_player ? state.white : state.black;
    player |= move; // Place the piece for the current player
    // Flip opponent's pieces in each direction
    uint64_t empty_squares = ~(state.black | state.white);
    flip_pieces(move, player, opponent, empty_squares);
    // Check for full board
    if (~(state.black + state.white) == 0) {
        game_over = true; // No empty squares left, game is over
    }
    // Switch current player
    current_player = !current_player;
    // Update legal moves and check for game over
    update_legal_moves();
    if (legal_moves == 0) {
        // Pass back turn
        current_player = !current_player; // Switch back to the other player
        update_legal_moves();
        if (legal_moves == 0) {
            game_over = true; // If both players have no legal moves, the game is over
        }
    }
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

uint64_t Board::legal_moves_edges(const uint64_t& empty, const uint64_t& player, const uint64_t& opponent) const {
    uint64_t moves = 0;
    // Directional masks to prevent wrapping
    const uint64_t notA = 0xfefefefefefefefeULL;
    const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;

    // North
    uint64_t mask = opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t temp = mask & (player << 8);
    for (int i = 0; i < 5; i++) temp |= mask & (temp << 8);
    moves |= empty & (temp << 8);

    // South
    temp = mask & (player >> 8);
    for (int i = 0; i < 5; i++) temp |= mask & (temp >> 8);
    moves |= empty & (temp >> 8);

    // East
    mask = opponent & notH;
    temp = mask & (player << 1);
    for (int i = 0; i < 5; i++) temp |= mask & (temp << 1);
    moves |= empty & (temp << 1);

    // West
    mask = opponent & notA;
    temp = mask & (player >> 1);
    for (int i = 0; i < 5; i++) temp |= mask & (temp >> 1);
    moves |= empty & (temp >> 1);

    return moves;
}

uint64_t Board::legal_moves_diagonals(const uint64_t& empty_squares, const uint64_t& player, const uint64_t& opponent) const {
    uint64_t moves = 0;
    const uint64_t notA = 0xfefefefefefefefeULL;
    const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;

    // Northeast (up-right)
    uint64_t mask = opponent & notH;
    uint64_t temp = mask & (player << 9);
    for (int i = 0; i < 5; i++) temp |= mask & (temp << 9);
    moves |= empty_squares & (temp << 9);

    // Northwest (up-left)
    mask = opponent & notA;
    temp = mask & (player << 7);
    for (int i = 0; i < 5; i++) temp |= mask & (temp << 7);
    moves |= empty_squares & (temp << 7);

    // Southeast (down-right)
    mask = opponent & notH;
    temp = mask & (player >> 7);
    for (int i = 0; i < 5; i++) temp |= mask & (temp >> 7);
    moves |= empty_squares & (temp >> 7);

    // Southwest (down-left)
    mask = opponent & notA;
    temp = mask & (player >> 9);
    for (int i = 0; i < 5; i++) temp |= mask & (temp >> 9);
    moves |= empty_squares & (temp >> 9);

    return moves;
}

void Board::flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent, const uint64_t&) {
    // Masks for board edges
    const uint64_t row1 = 0x00000000000000FFULL;
    const uint64_t row2 = 0x000000000000FF00ULL;
    const uint64_t row7 = 0x00FF000000000000ULL;
    const uint64_t row8 = 0xFF00000000000000ULL;
    const uint64_t notA = 0xfefefefefefefefeULL;
    const uint64_t notAB = 0xfcfcfcfcfcfcfcfcULL;
    const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
    const uint64_t notGH = 0x3f3f3f3f3f3f3f3fULL;

    struct Dir {
        int shift;
        uint64_t skip_mask; // If move & skip_mask, skip this direction
        uint64_t border_mask; // Used to prevent wrap during shifting
    };

    // Directions: N, S, E, W, NE, NW, SE, SW
    const Dir dirs[8] = {
        {  8, row1 | row2, 0xFFFFFFFFFFFFFFFFULL }, // N
        { -8, row7 | row8, 0xFFFFFFFFFFFFFFFFULL }, // S
        {  1, notH & notGH, notH },                 // E
        { -1, notA & notAB, notA },                 // W
        {  9, (row1 | row2) | (notH & notGH), notH }, // NE
        {  7, (row1 | row2) | (notA & notAB), notA }, // NW
        { -7, (row7 | row8) | (notH & notGH), notH }, // SE
        { -9, (row7 | row8) | (notA & notAB), notA }, // SW
    };

    for (const auto& d : dirs) {
        if (move & d.skip_mask) continue; // Skip if move is on/near the edge for this direction

        uint64_t flips = 0;
        uint64_t cur = move;
        while (true) {
            // Shift and mask to prevent wrapping
            if (d.shift > 0) {
                cur = (cur << d.shift) & d.border_mask;
            } else {
                cur = (cur >> -d.shift) & d.border_mask;
            }
            if (cur == 0) break;
            if (cur & opponent) {
                flips |= cur;
            } else if (cur & player) {
                player |= flips;
                opponent &= ~flips;
                break;
            } else {
                break;
            }
        }
    }
}