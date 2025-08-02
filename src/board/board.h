#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
#include <string>
#include <iostream>
#include <bit>

struct State {
    uint64_t black; // Bitboard for black pieces
    uint64_t white; // Bitboard for white pieces
};

class Board {
public:
    Board() {
        throw std::runtime_error("Only Board(State initial_state, bool current_player) constructor can be used"); // Default constructor should not be used
    }
    Board(State initial_state, bool current_player);
    Board(State state, bool current_player, uint64_t legal_moves, bool game_over); // Deep copy constructor
    void pretty_print() const;
    Board deep_copy() const;
    void make_move(uint64_t move);
    uint64_t get_legal_moves() const;
    bool get_current_player() const;
    bool is_game_over() const;
    const std::pair<const int, const int> get_scores() const;
private:
    State state;
    bool current_player; // true for black, false for white
    uint64_t legal_moves;
    bool game_over;
    void update_legal_moves();
    void generate_moves_right_shift(const uint64_t& player, const uint64_t& opponent, 
        const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift);
    void generate_moves_left_shift(const uint64_t& player, const uint64_t& opponent, 
        const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift);
    void detect_game_over();
    void flip_pieces(const uint64_t& move, uint64_t& player, uint64_t& opponent);
    void flip_pieces_right_shift(const uint64_t& move, uint64_t& player, uint64_t& opponent, 
        const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift);
    void flip_pieces_left_shift(const uint64_t& move, uint64_t& player, uint64_t& opponent, 
        const uint64_t initial_mask, const uint64_t wrap_mask, const unsigned int shift);
};

#endif // BOARD_H