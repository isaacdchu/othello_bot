#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

using othello_bot::MinimaxPlayer;

namespace {

othello::board::Board play(othello::player::Player& black, othello::player::Player& white) {
    othello::board::Board board;
    while (!board.game_over()) {
        auto& player = (board.turn() == othello::Color::BLACK) ? black : white;
        const uint64_t move = player.get_move(board);
        EXPECT_TRUE(board.move_valid(move));
        board.make_move(move);
    }
    return board;
}

// plain minimax with no pruning, scored from black's point of view,
// using the same leaf values as MinimaxPlayer
double reference_value(const othello::board::Board& board, int depth, const MinimaxPlayer::Heuristic& h) {
    const bool black = board.turn() == othello::Color::BLACK;
    if (board.game_over()) {
        const int diff = board.black_count() - board.white_count();
        if (diff > 0) {
            return MinimaxPlayer::WIN_SCORE + diff;
        }
        if (diff < 0) {
            return -MinimaxPlayer::WIN_SCORE + diff;
        }
        return 0.0;
    }
    if (depth == 0) {
        return black ? h(board) : -h(board);
    }
    double best = black ? -1e18 : 1e18;
    for (uint64_t rest = board.legal_moves(); rest != 0; rest &= rest - 1) {
        othello::board::Board child = board;
        child.make_move(rest & -rest);
        const double value = reference_value(child, child.game_over() ? 0 : depth - 1, h);
        best = black ? std::max(best, value) : std::min(best, value);
    }
    return best;
}

// first move (lowest bit first) reaching the best reference value
uint64_t reference_move(const othello::board::Board& board, int depth, const MinimaxPlayer::Heuristic& h) {
    const bool black = board.turn() == othello::Color::BLACK;
    uint64_t best_move = 0;
    double best = 0.0;
    for (uint64_t rest = board.legal_moves(); rest != 0; rest &= rest - 1) {
        const uint64_t move = rest & -rest;
        othello::board::Board child = board;
        child.make_move(move);
        const double value = reference_value(child, child.game_over() ? 0 : depth - 1, h);
        const double signed_value = black ? value : -value;
        if (best_move == 0 || signed_value > best) {
            best = signed_value;
            best_move = move;
        }
    }
    return best_move;
}

// checks MinimaxPlayer against plain minimax on positions from random games,
// including midgame passes and endgames
void expect_matches_reference(const MinimaxPlayer::Heuristic& h, int max_depth) {
    for (int depth = 1; depth <= max_depth; ++depth) {
        MinimaxPlayer player(depth, h);
        for (uint64_t seed = 0; seed < 4; ++seed) {
            othello::player::RandomPlayer random(seed);
            othello::board::Board board;
            while (!board.game_over()) {
                if (!std::has_single_bit(board.legal_moves())) {
                    ASSERT_EQ(player.get_move(board), reference_move(board, depth, h))
                        << "depth " << depth << "\n" << board.to_string();
                }
                board.make_move(random.get_move(board));
            }
        }
    }
}

} // namespace

TEST(MinimaxPlayer, RejectsBadDepth) {
    EXPECT_THROW(MinimaxPlayer(0), std::invalid_argument);
}

TEST(MinimaxPlayer, ReturnsLegalMoveFromStart) {
    MinimaxPlayer player(4);
    othello::board::Board board;
    const uint64_t move = player.get_move(board);
    EXPECT_TRUE(std::has_single_bit(move));
    EXPECT_TRUE(board.move_valid(move));
}

TEST(MinimaxPlayer, RejectsEmptyHeuristic) {
    EXPECT_THROW(MinimaxPlayer(4, nullptr), std::invalid_argument);
}

TEST(MinimaxPlayer, MatchesPlainMinimaxDiscDifference) {
    expect_matches_reference(MinimaxPlayer::disc_difference, 4);
}

TEST(MinimaxPlayer, MatchesPlainMinimaxMobility) {
    expect_matches_reference(MinimaxPlayer::mobility, 4);
}

TEST(MinimaxPlayer, MatchesPlainMinimaxCustomLambda) {
    // corners only
    constexpr uint64_t corners = 0x8100000000000081ULL;
    expect_matches_reference([](const othello::board::Board& b) {
        const double black = std::popcount(b.black() & corners) - std::popcount(b.white() & corners);
        return b.turn() == othello::Color::BLACK ? black : -black;
    }, 3);
}

TEST(MinimaxPlayer, HeuristicsFromSideToMove) {
    othello::board::Board board;
    // start position is symmetric
    EXPECT_EQ(MinimaxPlayer::disc_difference(board), 0.0);
    EXPECT_EQ(MinimaxPlayer::mobility(board), 0.0);

    // black d3 flips d4: black has 4 discs, white has 1; white to move
    // white's only disc is e5, so it can play c3, e3 or c5
    board.make_move(othello::util::parse_move("d3"));
    EXPECT_EQ(MinimaxPlayer::disc_difference(board), -3.0);
    const int white_moves = std::popcount(board.legal_moves(othello::Color::WHITE));
    const int black_moves = std::popcount(board.legal_moves(othello::Color::BLACK));
    EXPECT_EQ(white_moves, 3);
    EXPECT_EQ(MinimaxPlayer::mobility(board), white_moves - black_moves);
}

TEST(MinimaxPlayer, PruningVisitsFewerNodes) {
    MinimaxPlayer player(6);
    othello::board::Board board;
    player.get_move(board);
    // a full depth 6 tree from the start position has 10^4+ leaves
    EXPECT_GT(player.nodes(), 0u);
    EXPECT_LT(player.nodes(), 8'000u);
}

TEST(MinimaxPlayer, BeatsRandomMostly) {
    int wins = 0;
    constexpr int games = 10;
    for (int g = 0; g < games; ++g) {
        MinimaxPlayer minimax(6);
        othello::player::RandomPlayer random(g);
        const bool minimax_black = (g % 2 == 0);
        const auto board = minimax_black ? play(minimax, random) : play(random, minimax);
        const auto [black, white] = board.score();
        if (minimax_black ? black > white : white > black) {
            ++wins;
        }
    }
    EXPECT_GE(wins, 8);
}
