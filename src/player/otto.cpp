#ifndef OTTO_H
#define OTTO_H

#include "../player/player.h"
#include "../mcts/mctsnode.h"
#include <chrono>

class Otto : public Player {
public:
    Otto(const std::string &name, bool player_color)
        : Player(name, player_color) {}
    virtual ~Otto() = default;
    uint64_t get_move(const Board &board) override {
        // Trivial move cases
        uint64_t legal_moves = board.get_legal_moves();
        if (__builtin_popcountll(legal_moves) == 1) {
            return legal_moves; // Only one legal move available
        }
        if (legal_moves == 0x0000102004080000ULL) {
            return 0x0000000000080000ULL; // First move of the game is symmetric, so we can return a fixed move (d3)
        }
        // Use MCTS to find the best move
        MCTSNode root(0, board, nullptr, get_player_color());
        root.initialize_children(); // Initialize children nodes based on legal moves
        unsigned int iterations = 0;
        unsigned int max_iterations = 1000000;
        unsigned int num_simulations = 10;
        float average_simulation_time = 0.0f;
        for (unsigned int i = 0; i < max_iterations; i++) {
            auto node = root.select();
            if (node == nullptr) break;
            for (unsigned int j = 0; j < num_simulations; j++) {
                auto simulate_start = std::chrono::high_resolution_clock::now();
                const float result = node->simulate();
                auto simulate_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> simulation_duration = simulate_end - simulate_start;
                average_simulation_time += simulation_duration.count();
                node->backpropagate(result);
            }
            iterations++;
        }
        std::cout << "MCTS used " << iterations << " iterations." << std::endl;
        std::cout << "Average simulation time: " << average_simulation_time / (iterations * num_simulations) << " seconds." << std::endl;
        // Find the best move from the root node (most visited child)
        const uint64_t best_move = root.get_best_move();
        if ((best_move & legal_moves) == 0) {
            std::string error_message = "Best move " + move_to_square(best_move) + " is not a legal move!";
            throw std::runtime_error(error_message);
        }
        return best_move; // Return the best move found
    }
};

#endif // OTTO_H
