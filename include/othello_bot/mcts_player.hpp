#ifndef MCTS_PLAYER_HPP
#define MCTS_PLAYER_HPP

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <random>
#include <unordered_map>

#include <othello/othello.hpp>

namespace othello_bot {

// Classical UCT Monte Carlo tree search with random playouts.
//
// The tree is stored as a transposition table keyed by position, so positions
// reached through different move orders share statistics, and the part of the
// search that is still reachable carries over from one move to the next.
//
// UCT = w_i / n_i + c * sqrt(ln(N_i) / n_i)
class MCTSPlayer : public othello::player::Player {
private:
    struct Stats {
        uint32_t visits = 0;
        // results from black's point of view: win = 1, draw = 0.5, loss = 0
        double black_score = 0.0;
    };

    static double black_result(const othello::board::Board& board);

    uint64_t random_move(uint64_t moves);

    double rollout(othello::board::Board board);

    void iterate(const othello::board::Board& root);

    static othello::board::Board after(othello::board::Board board, uint64_t move) {
        board.make_move(move);
        return board;
    }

    std::size_t iterations_;
    double c_;
    std::mt19937_64 rng_;
    // Board's == and std::hash compare the discs and side to move only
    std::unordered_map<othello::board::Board, Stats> table_;

public:
    // iterations: playouts per move
    // seed: rng seed; std::nullopt uses 0
    // c: UCT exploration constant
    explicit MCTSPlayer(
        std::size_t iterations = 10'000,
        std::optional<uint64_t> seed = std::nullopt,
        double c = std::numbers::sqrt2
    );

    uint64_t get_move(const othello::board::Board& board) override;

    void set_iterations(std::size_t iterations) {
        iterations_ = iterations;
    }

    std::size_t iterations() const {
        return iterations_;
    }

    // drops all stored search statistics
    void clear() {
        table_.clear();
    }

    std::size_t table_size() const {
        return table_.size();
    }
};

} // namespace othello_bot

#endif // MCTS_PLAYER_HPP
