#include <bit>
#include <stdexcept>
#include <utility>

#include <othello_bot/minimax_player.hpp>

namespace othello_bot {

namespace {

// (own, opponent) discs for the side to move
std::pair<uint64_t, uint64_t> sides(const othello::board::Board& board) {
    return (board.turn() == othello::Color::BLACK)
        ? std::pair{board.black(), board.white()}
        : std::pair{board.white(), board.black()};
}

} // namespace

MinimaxPlayer::MinimaxPlayer(int depth, Heuristic heuristic) :
Player(),
depth_(1)
{
    set_depth(depth);
    set_heuristic(std::move(heuristic));
}

void MinimaxPlayer::set_depth(int depth) {
    if (depth < 1) {
        throw std::invalid_argument("[MinimaxPlayer] depth must be at least 1");
    }
    depth_ = depth;
}

void MinimaxPlayer::set_heuristic(Heuristic heuristic) {
    if (!heuristic) {
        throw std::invalid_argument("[MinimaxPlayer] heuristic must not be empty");
    }
    heuristic_ = std::move(heuristic);
}

double MinimaxPlayer::disc_difference(const othello::board::Board& board) {
    const auto [own, opp] = sides(board);
    return std::popcount(own) - std::popcount(opp);
}

double MinimaxPlayer::mobility(const othello::board::Board& board) {
    const othello::Color side = board.turn();
    const othello::Color other = othello::board::Board::opponent(side);
    return std::popcount(board.legal_moves(side)) - std::popcount(board.legal_moves(other));
}

double MinimaxPlayer::terminal_score(const othello::board::Board& board, othello::Color side) {
    const int diff = (side == othello::Color::BLACK)
        ? board.black_count() - board.white_count()
        : board.white_count() - board.black_count();
    if (diff > 0) {
        return WIN_SCORE + diff;
    }
    if (diff < 0) {
        return -WIN_SCORE + diff;
    }
    return 0.0;
}

double MinimaxPlayer::search(const othello::board::Board& board, int depth, double alpha, double beta) {
    ++nodes_;
    const othello::Color side = board.turn();
    if (board.game_over()) {
        return terminal_score(board, side);
    }
    if (depth == 0) {
        return heuristic_(board);
    }

    double best = -INF;
    for (uint64_t rest = board.legal_moves(); rest != 0; rest &= rest - 1) {
        othello::board::Board child = board;
        child.make_move(rest & -rest);

        double value;
        if (child.game_over()) {
            // game_over leaves turn() unchanged, so score it directly for `side`
            ++nodes_;
            value = terminal_score(child, side);
        } else if (child.turn() == side) {
            // opponent had to pass: same side moves again, same window
            value = search(child, depth - 1, alpha, beta);
        } else {
            value = -search(child, depth - 1, -beta, -alpha);
        }

        if (value > best) {
            best = value;
        }
        if (best > alpha) {
            alpha = best;
        }
        if (alpha >= beta) {
            break;
        }
    }
    return best;
}

uint64_t MinimaxPlayer::get_move(const othello::board::Board& board) {
    const uint64_t moves = board.legal_moves();
    if (moves == 0) {
        throw std::logic_error("[MinimaxPlayer] no legal moves");
    }
    nodes_ = 0;
    if (std::has_single_bit(moves)) {
        return moves;
    }

    const othello::Color side = board.turn();
    uint64_t best_move = moves & -moves;
    double alpha = -INF;
    const double beta = INF;

    for (uint64_t rest = moves; rest != 0; rest &= rest - 1) {
        const uint64_t move = rest & -rest;
        othello::board::Board child = board;
        child.make_move(move);

        double value;
        if (child.game_over()) {
            ++nodes_;
            value = terminal_score(child, side);
        } else if (child.turn() == side) {
            value = search(child, depth_ - 1, alpha, beta);
        } else {
            value = -search(child, depth_ - 1, -beta, -alpha);
        }

        if (value > alpha) {
            alpha = value;
            best_move = move;
        }
    }
    return best_move;
}

} // namespace othello_bot
