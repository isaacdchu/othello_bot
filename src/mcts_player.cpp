#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <othello_bot/mcts_player.hpp>

namespace othello_bot {

MCTSPlayer::MCTSPlayer(std::size_t iterations, std::optional<uint64_t> seed, double c) :
Player(),
iterations_(iterations),
c_(c),
rng_(seed ? *seed : 0)
{
    //
}

double MCTSPlayer::black_result(const othello::board::Board& board) {
    const int margin = board.margin();
    if (margin > 0) {
        return 1.0;
    }
    if (margin < 0) {
        return 0.0;
    }
    return 0.5;
}

uint64_t MCTSPlayer::random_move(uint64_t moves) {
    std::uniform_int_distribution<int> dist(0, std::popcount(moves) - 1);
    for (int k = dist(rng_); k > 0; --k) {
        moves &= moves - 1; // clear lowest set bit
    }
    return moves & -moves; // isolate lowest remaining bit
}

double MCTSPlayer::rollout(othello::board::Board board) {
    while (!board.game_over()) {
        board.make_move(random_move(board.legal_moves()));
    }
    return black_result(board);
}

void MCTSPlayer::iterate(const othello::board::Board& root) {
    // positions visited this iteration, root first
    std::vector<othello::board::Board> path;
    path.reserve(64);
    path.push_back(root);

    othello::board::Board state = root;
    while (!state.game_over()) {
        const uint64_t moves = state.legal_moves();
        const bool black_to_move = state.turn() == othello::Color::BLACK;
        const double parent_visits = table_[state].visits;
        const double log_parent = std::log(parent_visits);

        // children not yet in the table get expanded first
        std::array<uint64_t, 64> unexpanded;
        int unexpanded_count = 0;

        othello::board::Board best = state;
        double best_uct = -std::numeric_limits<double>::infinity();

        for (uint64_t rest = moves; rest != 0; rest &= rest - 1) {
            const uint64_t move = rest & -rest;
            const othello::board::Board child = after(state, move);
            const auto it = table_.find(child);
            if (it == table_.end() || it->second.visits == 0) {
                unexpanded[unexpanded_count++] = move;
                continue;
            }
            if (unexpanded_count > 0) {
                // an unexpanded child will be chosen anyway
                continue;
            }
            const Stats& stats = it->second;
            const double n = stats.visits;
            const double black_mean = stats.black_score / n;
            const double mean = black_to_move ? black_mean : 1.0 - black_mean;
            const double uct = mean + c_ * std::sqrt(log_parent / n);
            if (uct > best_uct) {
                best_uct = uct;
                best = child;
            }
        }

        if (unexpanded_count > 0) {
            // expansion: add one new node, then play out from it
            std::uniform_int_distribution<int> dist(0, unexpanded_count - 1);
            state = after(state, unexpanded[dist(rng_)]);
            table_.try_emplace(state);
            path.push_back(state);
            break;
        }

        state = best;
        path.push_back(state);
    }

    const double result = state.game_over()
        ? black_result(state)
        : rollout(state);

    for (const othello::board::Board& position : path) {
        Stats& stats = table_[position];
        ++stats.visits;
        stats.black_score += result;
    }
}

uint64_t MCTSPlayer::get_move(const othello::board::Board& board) {
    const uint64_t moves = board.legal_moves();
    if (moves == 0) {
        throw std::logic_error("[MCTSPlayer] no legal moves");
    }
    if (std::has_single_bit(moves)) {
        return moves;
    }

    // positions with fewer discs than the root can never come up again
    const int root_discs = std::popcount(board.black() | board.white());
    std::erase_if(table_, [root_discs](const auto& entry) {
        return std::popcount(entry.first.black() | entry.first.white()) < root_discs;
    });

    table_.try_emplace(board);
    for (std::size_t i = 0; i < iterations_; ++i) {
        iterate(board);
    }

    // play the most visited child
    uint64_t best_move = moves & -moves;
    uint32_t best_visits = 0;
    for (uint64_t rest = moves; rest != 0; rest &= rest - 1) {
        const uint64_t move = rest & -rest;
        const auto it = table_.find(after(board, move));
        if (it != table_.end() && it->second.visits > best_visits) {
            best_visits = it->second.visits;
            best_move = move;
        }
    }
    return best_move;
}

} // namespace othello_bot
