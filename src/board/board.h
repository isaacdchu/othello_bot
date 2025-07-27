#ifndef BOARD_H
#define BOARD_H

#include "../utils/utils.h"

#include <cstdint>
#include <string>
#include <iostream>

struct State {
    uint64_t black; // Bitboard for black pieces
    uint64_t white; // Bitboard for white pieces
};

class Board {
public:
    Board() {
        throw std::runtime_error("Only Board(State initial_state) constructor can be used"); // Default constructor should not be used
    }
    Board(State initial_state);
    void pretty_print() const;
    Board deep_copy() const;
    void make_move(uint64_t move, bool for_player);
    uint64_t get_legal_moves() const;
    bool get_current_player() const;
    bool is_game_over() const;

private:
    State state;
    bool current_player; // true for black, false for white
    uint64_t legal_moves;
    bool game_over;
    void update_legal_moves();
};

#endif // BOARD_H