#include "board.h"
#define ONE uint64_t(1) // Guarantees that ONE is a 64-bit unsigned integer

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
    legal_moves = legal_moves_edges(empty_squares, player, opponent) | legal_moves_diagonals(empty_squares, player, opponent);
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
    const uint64_t fileA = 0x0101010101010101ULL;
    const uint64_t fileH = 0x8080808080808080ULL;
    const uint64_t row1  = 0x00000000000000FFULL;
    const uint64_t row8  = 0xFF00000000000000ULL;

    struct Dir {
        int shift;
        uint64_t skip_mask;
        uint64_t border_mask;
    };

    const Dir dirs[8] = {
        {  8, row8, 0xFFFFFFFFFFFFFFFFULL }, // N
        { -8, row1, 0xFFFFFFFFFFFFFFFFULL }, // S
        {  1, fileH, 0xfefefefefefefefeULL }, // E
        { -1, fileA, 0x7f7f7f7f7f7f7f7fULL }, // W
        {  9, fileH | row8, 0xfefefefefefefefeULL }, // NE
        {  7, fileA | row8, 0x7f7f7f7f7f7f7f7fULL }, // NW
        { -7, fileH | row1, 0xfefefefefefefefeULL }, // SE
        { -9, fileA | row1, 0x7f7f7f7f7f7f7f7fULL }, // SW
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