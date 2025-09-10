#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <iostream>

#include "cnn/cnn.hpp"

int main();

// Old parse_line function
std::pair<Tensor<8, 8, 3>, const int> old_parse_line(const std::string &line) {
    // Parses a line from the training data file
    // Assumes valid input format: "^[-XO]{4}\s[\d-]\d*$"
    // X represents black pieces, O represents white pieces, - represents empty squares
    // Evaluation label describes piece difference for black (higher is better for black)
    // First character is a1, the second is a2, and so on
    Tensor<8, 8, 3> tensor = Tensor<8, 8, 3>();
    State state = {0, 0};
    const int evaluation = std::stoi(line.substr(65)); // Extract evaluation label from the line
    for (int i = 0; i < 64; i++) {
        if (line[i] == '-') {
            continue;
        } else if (line[i] == 'X') {
            tensor.set(i % 8, i / 8, 0, 1.0f); // Set black piece
            state.black |= (uint64_t(1) << i); // Set bit for black piece
        } else if (line[i] == 'O') {
            tensor.set(i % 8, i / 8, 1, 1.0f); // Set white piece
            state.white |= (uint64_t(1) << i); // Set bit for white piece
        }
    }
    const uint64_t legal_moves = Board::get_legal_moves(state); // Set legal moves in the tensor
    for (int i = 0; i < 64; i++) {
        if (legal_moves & (uint64_t(1) << i)) {
            tensor.set(i % 8, i / 8, 2, 1.0f); // Set legal move
        }
    }
    return {tensor, evaluation}; // Return the parsed state and evaluation
}

// Old main function
int old_main() {
    const std::string model_path = "model/cnn_model.txt";
    for (int i = 0; i <= 9; i++) {
        std::string data_path = "data/000000" + std::to_string(i) + ".txt";
        std::string output_path = "data/0" + std::to_string(i) + ".txt";
        std::cout << "Processing file: " << data_path << std::endl;
        std::string line;
        std::ifstream data(data_path); // Open the file here
        std::ofstream output(output_path); // Open the output file
        if (!data) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        while (std::getline(data, line)) {
            auto [tensor, evaluation] = old_parse_line(line);
            output << tensor.to_string() + " " + std::to_string(evaluation) << std::endl;
        }
    }
    for (int i = 10; i <= 25; i++) {
        std::string data_path = "data/00000" + std::to_string(i) + ".txt";
        std::string output_path = "data/" + std::to_string(i) + ".txt";
        std::cout << "Processing file: " << data_path << std::endl;
        std::string line;
        std::ifstream data(data_path); // Open the file here
        std::ofstream output(output_path); // Open the output file
        if (!data) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        while (std::getline(data, line)) {
            auto [tensor, evaluation] = old_parse_line(line);
            output << tensor.to_string() + " " + std::to_string(evaluation) << std::endl;
        }
    }
    return 0;
}

#endif // TRAIN_HPP