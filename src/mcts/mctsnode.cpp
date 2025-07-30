#ifndef MCTSNODE_H
#define MCTSNODE_H

#include "mctsnode.h"

MCTSNode::MCTSNode(const uint64_t move_to_get_here, const Board &board, MCTSNode *parent, const bool root_player)
    : move_to_get_here(move_to_get_here), board(board), parent(parent), root_player(root_player), visits(0), value(0.0f) {
    // Constructor initializes the node with the given parameters
    children_initialized = false; // Children are not initialized yet
}

void MCTSNode::initialize_children() {
    if (children_initialized) {
        return; // Children are already initialized, no need to do it again
    }
    children_initialized = true; // Mark children as initialized
    uint64_t temp_legal_moves = board.get_legal_moves(); // Get legal moves for the current board state
    if (temp_legal_moves == 0) {
        children.reserve(0); // Reserve no space for children
    } else {
        children.reserve(__builtin_popcountll(temp_legal_moves)); // Reserve space for children based on the number of legal moves
        // Populate all children nodes
        while (temp_legal_moves > 0) {
            uint64_t move = temp_legal_moves & -temp_legal_moves; // Get the lowest bit set (first legal move)
            temp_legal_moves &= ~move; // Remove this move from the legal moves
            Board child_board = board.deep_copy(); // Create a deep copy of the board
            child_board.make_move(move); // Make the move on the copied board
            children.emplace_back(std::make_unique<MCTSNode>(move, child_board, this, root_player)); // Create a new child node
        }
    }
}

MCTSNode* MCTSNode::select() {
    // Implementation for selecting a child node based on UCT value
    // Don't try to select if there are no children (i.e., game over)
    if (children.empty()) {
        return nullptr; // No children to select from
    }
    // Prioritize selecting unvisited nodes first
    for (const auto &child : children) {
        if (child->get_visits() == 0) {
            return child.get();
        }
    }
    // We have no unvisited children, so we are fully expanded
    // Select the child with the highest UCT value and recurse to find an unvisited node
    MCTSNode *best_child = nullptr;
    float best_uct = -1.0f; // UCT is always non-negative, so start with a negative value
    for (const auto &child : children) {
        float uct_value = child->get_uct();
        if (uct_value > best_uct) {
            best_uct = uct_value;
            best_child = child.get(); // Keep the raw pointer to the best child
        }
    }
    best_child->initialize_children(); // Ensure the best child has its children initialized
    return best_child->select(); // Recur on the best child to find an unvisited node
}

float MCTSNode::simulate() {
    // Returns the result of a random simulation from this node
    // Returned value is the difference in scores for the root player (higher is better for root player)
    Board simulation_board = board.deep_copy();
    uint64_t legal_moves = simulation_board.get_legal_moves();
    while (simulation_board.is_game_over() == false && legal_moves != 0) {
        // Randomly select a legal move and apply it
        uint64_t move = legal_moves & -legal_moves; // Get the lowest bit set (first legal move)
        simulation_board.make_move(move);
        legal_moves = simulation_board.get_legal_moves();
    }
    // Return the result of the simulation
    const auto scores = simulation_board.get_scores();
    if (root_player) {
        return static_cast<float>(scores.first - scores.second); // Root player is black
    } else {
        return static_cast<float>(scores.second - scores.first); // Root player is white
    }
}

void MCTSNode::backpropagate(const float result) {
    // Backpropagation logic to update the node's value and visits
    // Adds "result" to the node's value and increments visits
    // This continues up the tree until (and including) the root node
    value += result;
    visits++;
    if (parent != nullptr) {
        parent->backpropagate(result); // Propagate the result up to the parent node
    }
}

uint64_t MCTSNode::get_best_move() const {
    // Find the child with the highest visit count
    uint64_t best_move = 0;
    unsigned int max_visits = 0;
    for (const auto &child : children) {
        if (child->get_visits() > max_visits) {
            max_visits = child->get_visits();
            best_move = child->move_to_get_here;
        }
    }
    return best_move; // Return the move leading to the best child
}

#endif // MCTSNODE_H