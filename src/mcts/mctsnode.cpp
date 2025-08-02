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
        return; // No legal moves available, no children to create
    }
    children.reserve(__builtin_popcountll(temp_legal_moves)); // Reserve space for children based on the number of legal moves
    // Populate all children nodes
    while (temp_legal_moves > 0) {
        uint64_t move = temp_legal_moves & -temp_legal_moves; // Get the lowest bit set (first legal move)
        temp_legal_moves &= ~move; // Remove this move from the legal moves
        Board child_board = board.deep_copy(); // Create a deep copy of the board
        child_board.make_move(move); // Make the move on the copied board
        children.emplace_back(std::make_unique<MCTSNode>(move, child_board, this, root_player)); // Create a new child node
    }
    static thread_local std::mt19937 gen(std::random_device{}());
    std::shuffle(children.begin(), children.end(), gen); // Shuffle children for randomness
}

MCTSNode* MCTSNode::select() {
    // Implementation for selecting a child node based on UCT value
    // Don't try to select if there are no children (i.e., game over)
    if (board.is_game_over()) {
        return nullptr; // No children to select from
    }
    // Select the child with the highest UCT value and recurse to find an unvisited node
    MCTSNode *best_child = nullptr;
    float best_uct = std::numeric_limits<float>::lowest();
    float uct_value;
    for (const auto &child : children) {
        // Prioritize selecting unvisited nodes first
        if (child->get_visits() == 0) {
            return child.get();
        }
        // Get highest UCT value among all children
        uct_value = child->get_uct(1.5f);
        if (uct_value > best_uct) {
            best_uct = uct_value;
            best_child = child.get(); // Keep the raw pointer to the best child
        }
    }
    best_child->initialize_children(); // Ensure the best child has its children initialized
    // "best_child" is visited, so recur on the best child to find an unvisited node
    return best_child->select();
}

float MCTSNode::simulate() {
    // Returns the result of a random simulation from this node
    Board simulation_board = board.deep_copy();
    uint64_t legal_moves = simulation_board.get_legal_moves();
    // Array that holds bit indices of legal moves
    static thread_local std::mt19937 gen(std::random_device{}());
    while (legal_moves > 0) {
        int num_moves = __builtin_popcountll(legal_moves);
        std::uniform_int_distribution<> dis(0, num_moves - 1);
        int move_index = dis(gen);
        int idx = 0;
        int chosen_bit = -1; // Should be set to a valid bit index during the loop
        for (int b = 0; b < 64; b++) {
            if (!(legal_moves & (uint64_t(1ULL) << b))) continue; // Skip bits that are not set in legal_moves
            if (idx == move_index) {
                // Found the chosen move index
                chosen_bit = b;
                break;
            }
            idx++;
        }
        uint64_t move = uint64_t(1ULL) << chosen_bit;
        // Debugging check to ensure the move is legal
        // if ((move & legal_moves) == 0) {
        //     std::string error_message = "Randomly selected move " + move_to_square(move) + " is not a legal move!";
        //     throw std::runtime_error(error_message);
        // }
        simulation_board.make_move(move);
        legal_moves = simulation_board.get_legal_moves();
    }
    // Return the result of the simulation
    const auto scores = simulation_board.get_scores();
    if (scores.first == scores.second) return 0;
    if (root_player) {
        // Root player is black
        return (scores.first > scores.second) ? 1 : -1;
    } else {
        // Root player is white
        return (scores.second > scores.first) ? 1 : -1;
    }
}

void MCTSNode::backpropagate(const float result) {
    // Backpropagation logic to update the node's value and visits
    // Adds "result" to the node's value and increments visits
    // This continues up the tree until (and including) the root node
    MCTSNode* node = this;
    while (node != nullptr) {
        node->value += result;
        node->visits++;
        node = node->parent;
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