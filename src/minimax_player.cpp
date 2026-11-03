#include <algorithm>
#include <array>
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

// static move ordering priority of each square, indexed by bit. The table is
// symmetric under every rotation and reflection, so the bit layout doesn't matter.
// corners first, then edges and the centre, the squares next to corners last
constexpr std::array<int8_t, 64> SQUARE_PRIORITY = {
     4, -3,  2,  2,  2,  2, -3,  4,
    -3, -4, -1, -1, -1, -1, -4, -3,
     2, -1,  1,  0,  0,  1, -1,  2,
     2, -1,  0,  1,  1,  0, -1,  2,
     2, -1,  0,  1,  1,  0, -1,  2,
     2, -1,  1,  0,  0,  1, -1,  2,
    -3, -4, -1, -1, -1, -1, -4, -3,
     4, -3,  2,  2,  2,  2, -3,  4,
};

} // namespace

MinimaxPlayer::MinimaxPlayer(int depth, Heuristic heuristic, int table_bits) :
Player(),
depth_(1),
table_mask_(0)
{
    set_depth(depth);
    set_heuristic(std::move(heuristic));
    if (table_bits < 0 || table_bits > 30) {
        throw std::invalid_argument("[MinimaxPlayer] table_bits must be in [0, 30]");
    }
    table_.resize(std::size_t{1} << table_bits);
    table_mask_ = table_.size() - 1;
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

void MinimaxPlayer::set_time_limit(std::optional<Clock::duration> limit) {
    if (limit && *limit <= Clock::duration::zero()) {
        throw std::invalid_argument("[MinimaxPlayer] time limit must be positive");
    }
    time_limit_ = limit;
}

void MinimaxPlayer::clear_table() {
    std::fill(table_.begin(), table_.end(), Entry{});
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
    const int diff = (side == othello::Color::BLACK) ? board.margin() : -board.margin();
    if (diff > 0) {
        return WIN_SCORE + diff;
    }
    if (diff < 0) {
        return -WIN_SCORE + diff;
    }
    return 0.0;
}

int MinimaxPlayer::order_moves(uint64_t moves, uint8_t first, uint64_t* out) {
    int n = 0;
    for (uint64_t rest = moves; rest != 0; rest &= rest - 1) {
        out[n++] = rest & -rest;
    }
    // stable, so equal priorities keep lowest-bit-first order
    std::stable_sort(out, out + n, [](uint64_t a, uint64_t b) {
        return SQUARE_PRIORITY[std::countr_zero(a)] > SQUARE_PRIORITY[std::countr_zero(b)];
    });
    if (first != NO_MOVE) {
        const uint64_t move = uint64_t{1} << first;
        uint64_t* it = std::find(out, out + n, move);
        if (it != out + n) {
            std::rotate(out, it, it + 1);
        }
    }
    return n;
}

MinimaxPlayer::Entry& MinimaxPlayer::slot(const othello::board::Board& board) {
    return table_[board.hash() & table_mask_];
}

void MinimaxPlayer::store(
    const othello::board::Board& board,
    int depth,
    double value,
    Bound bound,
    uint8_t best_move
) {
    if (aborted_) {
        // the value of an interrupted search is meaningless
        return;
    }
    Entry& e = slot(board);
    // keep a deeper entry from this search; anything else is replaced
    if (e.depth > depth && e.generation == generation_ && !(e.board == board)) {
        return;
    }
    e = Entry{
        .board = board,
        .value = value,
        .generation = generation_,
        .depth = static_cast<int16_t>(depth),
        .bound = bound,
        .best_move = best_move,
    };
}

bool MinimaxPlayer::out_of_time() {
    if (aborted_) {
        return true;
    }
    // reading the clock is cheap next to a node, but not free
    if (deadline_ && (nodes_ & 63) == 0 && Clock::now() >= *deadline_) {
        aborted_ = true;
    }
    return aborted_;
}

double MinimaxPlayer::search(const othello::board::Board& board, int depth, double alpha, double beta) {
    ++nodes_;
    if (out_of_time()) {
        return 0.0;
    }
    const othello::Color side = board.turn();
    if (board.game_over()) {
        return terminal_score(board, side);
    }

    uint8_t table_move = NO_MOVE;
    {
        // a reference into the table is only valid until the next store
        const Entry& e = slot(board);
        if (e.depth >= 0 && e.board == board) {
            table_move = e.best_move;
            if (e.generation == generation_ && e.depth == depth) {
                if (
                    e.bound == Bound::EXACT ||
                    (e.bound == Bound::LOWER && e.value >= beta) ||
                    (e.bound == Bound::UPPER && e.value <= alpha)
                ) {
                    return e.value;
                }
            }
        }
    }

    if (depth == 0) {
        // leaves are cached too: with a network as heuristic they are the expensive part
        const double value = heuristic_(board);
        store(board, 0, value, Bound::EXACT, NO_MOVE);
        return value;
    }

    std::array<uint64_t, 64> moves;
    const int n = order_moves(board.legal_moves(), table_move, moves.data());

    const double original_alpha = alpha;
    double best = -INF;
    uint8_t best_move = NO_MOVE;
    for (int i = 0; i < n; ++i) {
        othello::board::Board child = board;
        child.make_move(moves[i]);

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
        if (aborted_) {
            return 0.0;
        }

        if (value > best) {
            best = value;
            best_move = static_cast<uint8_t>(std::countr_zero(moves[i]));
        }
        if (best > alpha) {
            alpha = best;
        }
        if (alpha >= beta) {
            break;
        }
    }

    // fail-soft: best is exact inside the window, a bound outside it
    const Bound bound = (best <= original_alpha) ? Bound::UPPER
        : (best >= beta) ? Bound::LOWER
        : Bound::EXACT;
    store(board, depth, best, bound, best_move);
    return best;
}

uint64_t MinimaxPlayer::get_move(const othello::board::Board& board) {
    const uint64_t legal = board.legal_moves();
    if (legal == 0) {
        throw std::logic_error("[MinimaxPlayer] no legal moves");
    }
    const Clock::time_point start = Clock::now();
    nodes_ = 0;
    last_value_.reset();
    last_depth_ = 0;
    aborted_ = false;
    deadline_.reset();
    if (std::has_single_bit(legal)) {
        return legal;
    }
    ++generation_;

    const othello::Color side = board.turn();
    uint8_t table_move = NO_MOVE;
    if (const Entry& e = slot(board); e.depth >= 0 && e.board == board) {
        table_move = e.best_move;
    }
    std::array<uint64_t, 64> moves;
    const int n = order_moves(legal, table_move, moves.data());
    uint64_t best_move = moves[0];

    for (int depth = 1; depth <= depth_; ++depth) {
        // depth 1 always completes, so there is always a searched move
        if (depth > 1 && time_limit_) {
            deadline_ = start + *time_limit_;
            if (Clock::now() >= *deadline_) {
                break;
            }
        }

        // beta is infinite at the root, so every child is searched and alpha
        // ends as the exact value of the best one
        double alpha = -INF;
        int iteration_best = 0;
        for (int i = 0; i < n; ++i) {
            othello::board::Board child = board;
            child.make_move(moves[i]);

            double value;
            if (child.game_over()) {
                ++nodes_;
                value = terminal_score(child, side);
            } else if (child.turn() == side) {
                value = search(child, depth - 1, alpha, INF);
            } else {
                value = -search(child, depth - 1, -INF, -alpha);
            }
            if (aborted_) {
                break;
            }
            if (value > alpha) {
                alpha = value;
                iteration_best = i;
            }
        }
        if (aborted_) {
            // keep the previous iteration's result
            break;
        }

        best_move = moves[iteration_best];
        last_value_ = alpha;
        last_depth_ = depth;
        store(board, depth, alpha, Bound::EXACT, static_cast<uint8_t>(std::countr_zero(best_move)));
        // search this iteration's best move first in the next one
        std::rotate(moves.begin(), moves.begin() + iteration_best, moves.begin() + iteration_best + 1);
    }
    return best_move;
}

} // namespace othello_bot
