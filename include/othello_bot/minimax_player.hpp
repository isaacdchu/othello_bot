#ifndef MINIMAX_PLAYER_HPP
#define MINIMAX_PLAYER_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <othello/othello.hpp>

namespace othello_bot {

// Depth-limited minimax (negamax form) with alpha-beta pruning, iterative
// deepening, a transposition table and move ordering.
//
// Leaves are scored by a heuristic that is passed in. Finished games score
// far beyond any heuristic value, so a forced win is always preferred and a
// bigger winning margin beats a smaller one.
//
// Passes are handled by the board: when make_move leaves the same side to
// move, the child is searched from the same point of view instead of negated.
//
// The value of every completed search is exactly the plain fixed-depth
// minimax value; the optimizations only change how many nodes are visited
// and, among equally good moves, which one is returned:
//   - iterative deepening searches depth 1, 2, ..., depth(), and orders each
//     iteration's root moves by the previous one's result. With a time limit
//     the deepest completed iteration is used.
//   - the transposition table stores values (with alpha-beta bounds) and best
//     moves. Values are only reused within one get_move call and at the same
//     remaining depth, so a heuristic that changes between calls (e.g. a
//     network being trained) never sees stale values. Best moves from earlier
//     calls are still used for move ordering.
//   - other moves are tried in a static order: corners first, squares next to
//     corners last.
class MinimaxPlayer : public othello::player::Player {
public:
    // Scores a position that is not over, from the point of view of the side
    // to move (board.turn()): higher is better for that side. A heuristic
    // should be zero-sum: scoring the same discs with the other side to move
    // should give the negated value. Keep values well below WIN_SCORE.
    // It must be deterministic during a get_move call.
    using Heuristic = std::function<double(const othello::board::Board&)>;

    using Clock = std::chrono::steady_clock;

    // finished games score WIN_SCORE plus the disc margin, from the winner's side
    static constexpr double WIN_SCORE = 1e9;

    // 2^16 entries of 48 bytes, about 3 MB
    static constexpr int DEFAULT_TABLE_BITS = 16;

private:
    static constexpr double INF = 1e18;
    static constexpr uint8_t NO_MOVE = 64;

    enum class Bound : uint8_t {
        EXACT,
        LOWER,  // value >= stored value
        UPPER,  // value <= stored value
    };

    struct Entry {
        othello::board::Board board;
        double value = 0.0;
        uint32_t generation = 0;
        // remaining depth the value was searched to; -1 for an empty slot
        int16_t depth = -1;
        Bound bound = Bound::EXACT;
        // bit index of the best move found, or NO_MOVE
        uint8_t best_move = NO_MOVE;
    };

    // score of a finished game from the point of view of `side`
    static double terminal_score(const othello::board::Board& board, othello::Color side);

    // value of board from the point of view of the side to move
    double search(const othello::board::Board& board, int depth, double alpha, double beta);

    // writes the moves in search order into `out` and returns how many there are
    static int order_moves(uint64_t moves, uint8_t first, uint64_t* out);

    Entry& slot(const othello::board::Board& board);

    void store(const othello::board::Board& board, int depth, double value, Bound bound, uint8_t best_move);

    // true once the deadline has passed (checked every few nodes)
    bool out_of_time();

    int depth_;
    Heuristic heuristic_;
    std::optional<Clock::duration> time_limit_;
    std::vector<Entry> table_;
    uint64_t table_mask_;
    // incremented on every get_move; values from other generations are not reused
    uint32_t generation_ = 0;

    // state of the current get_move
    std::optional<Clock::time_point> deadline_;
    bool aborted_ = false;

    std::size_t nodes_ = 0;
    std::optional<double> last_value_;
    int last_depth_ = 0;

public:
    // own discs minus opponent discs
    static double disc_difference(const othello::board::Board& board);

    // own legal moves minus opponent legal moves
    static double mobility(const othello::board::Board& board);

    // depth: maximum plies searched per move (at least 1)
    // heuristic: scores leaf positions; must not be empty
    // table_bits: the transposition table has 2^table_bits entries (0 to 30)
    explicit MinimaxPlayer(
        int depth = 6,
        Heuristic heuristic = disc_difference,
        int table_bits = DEFAULT_TABLE_BITS
    );

    uint64_t get_move(const othello::board::Board& board) override;

    void set_depth(int depth);

    int depth() const {
        return depth_;
    }

    void set_heuristic(Heuristic heuristic);

    // wall-clock budget per get_move, or std::nullopt (the default) to always
    // search to depth(). The search may overrun it by a few nodes. Results then
    // depend on machine speed, so leave it unset where reproducibility matters.
    void set_time_limit(std::optional<Clock::duration> limit);

    std::optional<Clock::duration> time_limit() const {
        return time_limit_;
    }

    // forgets every stored position, including the best moves used for ordering
    void clear_table();

    // nodes visited during the last get_move, over all iterations
    std::size_t nodes() const {
        return nodes_;
    }

    // value of the position given to the last get_move, from the point of view
    // of its side to move, on the same scale as the heuristic (finished games
    // are +-(WIN_SCORE + margin)), at last_depth(). Empty before the first
    // call, and when the last call had a single legal move, since no search
    // was run then.
    std::optional<double> last_value() const {
        return last_value_;
    }

    // depth of the deepest completed iteration in the last get_move (0 if no
    // search was run); below depth() only when the time limit ran out
    int last_depth() const {
        return last_depth_;
    }
};

} // namespace othello_bot

#endif // MINIMAX_PLAYER_HPP
