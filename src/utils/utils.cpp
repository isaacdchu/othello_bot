#include "utils.h"

std::string move_to_square(uint64_t move) {
    if (move == 0)
        return "Invalid move";
    int index = __builtin_ctzll(move); // Get the index of the least significant bit
    char column = 'a' + (index % 8);
    int row = (index / 8) + 1;
    return std::string(1, column) + std::to_string(row);
}

uint64_t square_to_move(const std::string& square) {
    if (square.length() != 2)
        return 0; // Invalid input
    char column = square[0];
    int row = square[1] - '0';
    if (column < 'a' || column > 'h' || row < 1 || row > 8)
        return 0; // Invalid input
    int index = (row - 1) * 8 + (column - 'a');
    return 1ULL << index; // Return the move as a bitboard
}