#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <iostream>
#include <string>
#include <limits>

std::string move_to_square(uint64_t move);
uint64_t square_to_move(const std::string& square);
void print_legal_moves(const uint64_t legal_moves);

#endif // UTILS_H