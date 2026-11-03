#include <algorithm>
#include <bit>
#include <chrono>
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

// reference value of the position from the point of view of its side to move
double reference_value_for_mover(const othello::board::Board& board, int depth, const MinimaxPlayer::Heuristic& h) {
    const double black_value = reference_value(board, depth, h);
    return board.turn() == othello::Color::BLACK ? black_value : -black_value;
}

// reference value of playing `move`, from the point of view of the side playing it
double reference_move_value(
    const othello::board::Board& board,
    uint64_t move,
    int depth,
    const MinimaxPlayer::Heuristic& h
) {
    othello::board::Board child = board;
    child.make_move(move);
    const double black_value = reference_value(child, child.game_over() ? 0 : depth - 1, h);
    return board.turn() == othello::Color::BLACK ? black_value : -black_value;
}

// checks MinimaxPlayer against plain minimax on positions from random games,
// including midgame passes and endgames: the value must match exactly, and the
// chosen move must be one of the best (ties may go to any of them). One player
// is reused across positions, so the transposition table carries over; tiny
// tables force constant collisions.
void expect_matches_reference(const MinimaxPlayer::Heuristic& h, int max_depth) {
    for (const int table_bits : {0, 4, MinimaxPlayer::DEFAULT_TABLE_BITS}) {
        for (int depth = 1; depth <= max_depth; ++depth) {
            MinimaxPlayer player(depth, h, table_bits);
            for (uint64_t seed = 0; seed < 4; ++seed) {
                othello::player::RandomPlayer random(seed);
                othello::board::Board board;
                while (!board.game_over()) {
                    if (!std::has_single_bit(board.legal_moves())) {
                        const uint64_t move = player.get_move(board);
                        const double expected = reference_value_for_mover(board, depth, h);
                        ASSERT_TRUE(board.move_valid(move));
                        ASSERT_EQ(player.last_value(), expected)
                            << "depth " << depth << " table_bits " << table_bits << "\n"
                            << board.to_string();
                        ASSERT_EQ(reference_move_value(board, move, depth, h), expected)
                            << "depth " << depth << " table_bits " << table_bits << "\n"
                            << board.to_string();
                        ASSERT_EQ(player.last_depth(), depth);
                    }
                    board.make_move(random.get_move(board));
                }
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

TEST(MinimaxPlayer, LastValueEmptyWithoutSearch) {
    MinimaxPlayer player(2);
    EXPECT_FALSE(player.last_value().has_value());

    othello::board::Board board;
    player.get_move(board);
    EXPECT_TRUE(player.last_value().has_value());

    // play random games until a position with a single legal move comes up;
    // get_move skips the search there
    for (uint64_t seed = 0; seed < 50; ++seed) {
        othello::player::RandomPlayer random(seed);
        board = othello::board::Board();
        while (!board.game_over() && !std::has_single_bit(board.legal_moves())) {
            board.make_move(random.get_move(board));
        }
        if (!board.game_over()) {
            player.get_move(board);
            EXPECT_FALSE(player.last_value().has_value());
            return;
        }
    }
    GTEST_SKIP() << "no forced move found";
}

TEST(MinimaxPlayer, NoStaleValuesAfterHeuristicChange) {
    // the same position searched with one heuristic, then another: the table
    // must not hand back the first heuristic's values
    othello::player::RandomPlayer random(7);
    othello::board::Board board;
    for (int i = 0; i < 12; ++i) {
        board.make_move(random.get_move(board));
    }
    ASSERT_FALSE(std::has_single_bit(board.legal_moves()));
    MinimaxPlayer player(3, MinimaxPlayer::disc_difference);
    player.get_move(board);
    EXPECT_EQ(player.last_value(), reference_value_for_mover(board, 3, MinimaxPlayer::disc_difference));
    player.set_heuristic(MinimaxPlayer::mobility);
    player.get_move(board);
    EXPECT_EQ(player.last_value(), reference_value_for_mover(board, 3, MinimaxPlayer::mobility));
}

TEST(MinimaxPlayer, TimeLimitStopsEarly) {
    using namespace std::chrono_literals;
    MinimaxPlayer player(40);
    player.set_time_limit(20ms);
    othello::board::Board board;
    const auto start = std::chrono::steady_clock::now();
    const uint64_t move = player.get_move(board);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_TRUE(board.move_valid(move));
    EXPECT_GE(player.last_depth(), 1);
    EXPECT_LT(player.last_depth(), 40);
    EXPECT_TRUE(player.last_value().has_value());
    // generous: the limit is checked every few nodes
    EXPECT_LT(elapsed, 500ms);
}

TEST(MinimaxPlayer, RejectsBadOptions) {
    EXPECT_THROW(MinimaxPlayer(4, MinimaxPlayer::disc_difference, -1), std::invalid_argument);
    EXPECT_THROW(MinimaxPlayer(4, MinimaxPlayer::disc_difference, 31), std::invalid_argument);
    MinimaxPlayer player(4);
    EXPECT_THROW(player.set_time_limit(std::chrono::milliseconds(0)), std::invalid_argument);
}

TEST(MinimaxPlayer, TableReducesNodes) {
    othello::player::RandomPlayer random(3);
    othello::board::Board board;
    for (int i = 0; i < 16; ++i) {
        board.make_move(random.get_move(board));
    }
    MinimaxPlayer tiny(7, MinimaxPlayer::mobility, 0);
    MinimaxPlayer full(7, MinimaxPlayer::mobility);
    tiny.get_move(board);
    full.get_move(board);
    EXPECT_EQ(tiny.last_value(), full.last_value());
    EXPECT_LT(full.nodes(), tiny.nodes());
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
