#include <bit>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

namespace {

// plays a full game between two players and returns the final board
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

} // namespace

TEST(MCTSPlayer, ReturnsLegalMoveFromStart) {
    othello_bot::MCTSPlayer player(500, 1);
    othello::board::Board board;
    const uint64_t move = player.get_move(board);
    EXPECT_TRUE(std::has_single_bit(move));
    EXPECT_TRUE(board.move_valid(move));
}

TEST(MCTSPlayer, SameSeedSameMoves) {
    othello_bot::MCTSPlayer a(300, 42);
    othello_bot::MCTSPlayer b(300, 42);
    othello::board::Board board;
    while (!board.game_over()) {
        const uint64_t move = a.get_move(board);
        ASSERT_EQ(move, b.get_move(board));
        board.make_move(move);
    }
}

TEST(MCTSPlayer, PlaysFullGameAgainstRandom) {
    othello_bot::MCTSPlayer mcts(200, 7);
    othello::player::RandomPlayer random(7);
    const auto board = play(mcts, random);
    EXPECT_TRUE(board.game_over());
}

TEST(MCTSPlayer, BeatsRandomMostly) {
    int wins = 0;
    constexpr int games = 10;
    for (int g = 0; g < games; ++g) {
        othello_bot::MCTSPlayer mcts(1000, g);
        othello::player::RandomPlayer random(g);
        // alternate colors
        const auto board = (g % 2 == 0) ? play(mcts, random) : play(random, mcts);
        const auto [black, white] = board.score();
        const bool mcts_black = (g % 2 == 0);
        if (mcts_black ? black > white : white > black) {
            ++wins;
        }
    }
    EXPECT_GE(wins, 8);
}
