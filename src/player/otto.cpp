#ifndef OTTO_H
#define OTTO_H

#include "../player/player.h"
#include "../mcts/mctsnode.h"

class Otto : public Player {
public:
    Otto(const std::string &name, bool player_color)
        : Player(name, player_color) {}
    virtual ~Otto() = default;
    uint64_t get_move(const Board &board) override {
        std::cout << "Otto is thinking..." << std::endl;
        // Trivial move cases
        uint64_t legal_moves = board.get_legal_moves();
        if (legal_moves == 0) {
            return 0; // No legal moves available
        }
        if (__builtin_popcountll(legal_moves) == 1) {
            return legal_moves; // Only one legal move available
        }
        // Use MCTS to find the best move
        std::cout << "Initializing root" << std::endl;
        MCTSNode root(0, board, nullptr, get_player_color());
        root.initialize_children(); // Initialize children nodes based on legal moves
        std::cout << "Running MCTS loop" << std::endl;
        for (int i = 0; i < 1000; i++) {
            std::cout << "Iteration " << i + 1 << std::endl;
            std::cout << "Selecting node" << std::endl;
            auto node = root.select();
            if (node == nullptr) break; // No valid moves available
            std::cout << "Simulating node" << std::endl;
            const float result = node->simulate();
            std::cout << "Backpropagating result" << std::endl;
            node->backpropagate(result);
        }
        // Find the best move from the root node (most visited child)
        const uint64_t best_move = root.get_best_move();
        delete &root; // Clean up the root node
        return best_move; // Return the best move found
    }
};

#endif // OTTO_H
