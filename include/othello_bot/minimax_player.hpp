#ifndef MINIMAX_PLAYER_HPP
#define MINIMAX_PLAYER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>

#include <othello/othello.hpp>

namespace othello_bot {

// Depth-limited minimax (negamax form) with alpha-beta pruning.
//
// Leaves are scored by a heuristic that is passed in. Finished games score
// far beyond any heuristic value, so a forced win is always preferred and a
// bigger winning margin beats a smaller one.
//
// Passes are handled by the board: when make_move leaves the same side to
// move, the child is searched from the same point of view instead of negated.
class MinimaxPlayer : public othello::player::Player {
public:
    // Scores a position that is not over, from the point of view of the side
    // to move (board.turn()): higher is better for that side. A heuristic
    // should be zero-sum: scoring the same discs with the other side to move
    // should give the negated value. Keep values well below WIN_SCORE.
    using Heuristic = std::function<double(const othello::board::Board&)>;

private:
    static constexpr double INF = 1e18;

    // score of a finished game from the point of view of `side`
    static double terminal_score(const othello::board::Board& board, othello::Color side);

    // value of board from the point of view of the side to move
    double search(const othello::board::Board& board, int depth, double alpha, double beta);

    int depth_;
    Heuristic heuristic_;
    std::size_t nodes_ = 0;

public:

    // finished games score WIN_SCORE plus the disc margin, from the winner's side
    static constexpr double WIN_SCORE = 1e9;

    // own discs minus opponent discs
    static double disc_difference(const othello::board::Board& board);

    // own legal moves minus opponent legal moves
    static double mobility(const othello::board::Board& board);

    // depth: plies searched per move (at least 1)
    // heuristic: scores leaf positions; must not be empty
    explicit MinimaxPlayer(int depth = 6, Heuristic heuristic = disc_difference);

    uint64_t get_move(const othello::board::Board& board) override;

    void set_depth(int depth);

    int depth() const {
        return depth_;
    }

    void set_heuristic(Heuristic heuristic);

    // nodes visited during the last get_move
    std::size_t nodes() const {
        return nodes_;
    }
};

} // namespace othello_bot

#endif // MINIMAX_PLAYER_HPP
