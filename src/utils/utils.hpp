#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <string>
#include <limits>
#include <vector>

std::string move_to_square(uint64_t move);
uint64_t square_to_move(const std::string& square);
void print_legal_moves(const uint64_t legal_moves);
template <class T>
void print_vector(const std::vector<T>& vec) {
    std::cout << "[";
    for (std::size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

#endif // UTILS_HPP