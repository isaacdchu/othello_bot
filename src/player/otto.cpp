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
        // Trivial move cases
        uint64_t legal_moves = board.get_legal_moves();
        if (legal_moves == 0) {
            return 0; // No legal moves available
        }
        if (__builtin_popcountll(legal_moves) == 1) {
            return legal_moves; // Only one legal move available
        }
        // Use MCTS to find the best move
        MCTSNode root(0, board, nullptr, get_player_color());
        root.initialize_children(); // Initialize children nodes based on legal moves
        unsigned int i;
        for (i = 0; i < 2000; i++) {
            auto node = root.select();
            if (node == nullptr) break; // No valid moves available
            const float result = node->simulate();
            node->backpropagate(result);
        }
        std::cout << "MCTS completed after " << i << " iterations." << std::endl;
        // Find the best move from the root node (most visited child)
        const uint64_t best_move = root.get_best_move();
        return best_move; // Return the best move found
    }
};

#endif // OTTO_H
