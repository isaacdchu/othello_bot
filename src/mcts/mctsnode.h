#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include "../board/board.h"
#include "../utils/utils.h"
#include <vector>
#include <memory>

class MCTSNode {
public:
    MCTSNode(const uint64_t move_to_get_here, const Board &board, MCTSNode *parent, const bool root_player);
    ~MCTSNode() {
        // Destructor to clean up children nodes
        for (auto &child : children) {
            child.reset();
        }
        parent = nullptr; // Clear parent pointer
    }
    const uint64_t move_to_get_here; // The move that led from its parent to this node
    void initialize_children();
    MCTSNode* select();
    float simulate();
    void backpropagate(const float result);
    uint64_t get_best_move() const;
    unsigned int get_visits() const {
        return visits;
    }
private:
    Board board;
    MCTSNode *parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    const bool root_player; // true for black, false for white
    unsigned int visits;
    float value;
    float get_uct(float c = 1.41) const {
        if (visits == 0) return 0.0f; // Avoid division by zero
        unsigned int parent_visits = parent ? parent->get_visits() : 1; // Avoid log of zero
        return value / visits + c * sqrt(2 * log(parent_visits) / visits);
    }
};

#endif // MCTS_NODE_H
