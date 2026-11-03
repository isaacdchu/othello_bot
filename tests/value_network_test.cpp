#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

namespace enc = othello_bot::encoding;
using othello_bot::MinimaxPlayer;
using othello_bot::Sample;
using othello_bot::ValueNetwork;

TEST(Encoding, PlanesAreFromSideToMove) {
    othello::board::Board board;
    auto p = enc::planes(board);
    EXPECT_EQ(p.own, board.black());
    EXPECT_EQ(p.opp, board.white());
    EXPECT_EQ(p.legal, board.legal_moves());

    board.make_move(othello::util::parse_move("d3"));
    p = enc::planes(board);
    EXPECT_EQ(p.own, board.white());
    EXPECT_EQ(p.opp, board.black());
}

TEST(Encoding, PlanesOfConstructedBoard) {
    // white to move on a hand-built position
    const uint64_t black = 0x0000000810000000ULL;
    const uint64_t white = 0x0000001008000000ULL;
    const othello::board::Board board(black, white, othello::Color::WHITE);
    const auto p = enc::planes(board);
    EXPECT_EQ(p.own, white);
    EXPECT_EQ(p.opp, black);
    EXPECT_EQ(p.legal, board.legal_moves(othello::Color::WHITE));
}

TEST(Encoding, EncodeMatchesBits) {
    const othello::board::Board board;
    std::vector<float> x(enc::INPUT_SIZE);
    enc::encode(enc::planes(board), x);
    float total = 0.0f;
    for (float v : x) {
        total += v;
    }
    // 2 own discs, 2 opponent discs, 4 legal moves
    EXPECT_EQ(total, 8.0f);
    // d5 (row 4, col 3) is black's in the start position
    EXPECT_EQ(x[0 * 64 + 4 * 8 + 3], 1.0f);
}

TEST(Encoding, SymmetriesAreDistinctAndPreserveCounts) {
    // a1-ish corner square plus one neighbour: asymmetric, so all 8 images differ
    const uint64_t bb = (1ULL << 63) | (1ULL << 62);
    std::set<uint64_t> images;
    for (int s = 0; s < enc::SYMMETRIES; ++s) {
        const uint64_t t = enc::transform(bb, s);
        EXPECT_EQ(std::popcount(t), 2);
        images.insert(t);
    }
    EXPECT_EQ(images.size(), 8u);
    EXPECT_EQ(enc::transform(bb, 0), bb);
}

TEST(Encoding, SymmetriesMapLegalMovesToLegalMoves) {
    // the start position is symmetric under some transforms; check a later one
    othello::board::Board board;
    othello::player::RandomPlayer random(1);
    for (int i = 0; i < 10; ++i) {
        board.make_move(random.get_move(board));
    }
    const auto p = enc::planes(board);
    for (int s = 0; s < enc::SYMMETRIES; ++s) {
        const auto t = enc::transform(p, s);
        EXPECT_EQ(std::popcount(t.legal), std::popcount(p.legal));
        EXPECT_EQ(t.own & t.opp, 0u);
    }
}

TEST(ValueNetwork, EvaluatesInRange) {
    auto net = ValueNetwork::create(0, 8);
    const othello::board::Board board;
    const float v = net.evaluate(board);
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, -1.0f);
    EXPECT_LE(v, 1.0f);
}

TEST(ValueNetwork, CloneMatches) {
    auto net = ValueNetwork::create(3, 8);
    auto copy = net.clone();
    othello::board::Board board;
    board.make_move(othello::util::parse_move("d3"));
    EXPECT_EQ(net.evaluate(board), copy.evaluate(board));
}

TEST(ValueNetwork, CopyWeightsFrom) {
    auto a = ValueNetwork::create(3, 8);
    auto b = ValueNetwork::create(4, 8);
    othello::board::Board board;
    board.make_move(othello::util::parse_move("d3"));
    ASSERT_NE(a.evaluate(board), b.evaluate(board));
    b.copy_weights_from(a);
    EXPECT_EQ(a.evaluate(board), b.evaluate(board));
}

TEST(ValueNetwork, TrainingReducesLossOnFixedBatch) {
    auto net = ValueNetwork::create(1, 8);
    std::mt19937_64 rng(0);
    othello::player::RandomPlayer random(2);
    std::vector<Sample> batch;
    othello::board::Board board;
    while (batch.size() < 8 && !board.game_over()) {
        batch.push_back({enc::planes(board), (batch.size() % 2 == 0) ? 1.0f : -1.0f});
        board.make_move(random.get_move(board));
    }
    const float first = net.train_batch(batch, rng);
    float last = first;
    for (int i = 0; i < 50; ++i) {
        last = net.train_batch(batch, rng);
    }
    EXPECT_LT(last, first);
}

TEST(ValueNetwork, DrivesMinimax) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(2, net.heuristic());
    const othello::board::Board board;
    EXPECT_TRUE(board.move_valid(player.get_move(board)));
    EXPECT_GT(net.evaluations(), 0u);
}

TEST(SelfPlay, GameProducesLabelledSamples) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(1, net.heuristic());
    std::mt19937_64 rng(0);
    const auto samples = othello_bot::self_play::play_game(player, {}, rng);
    ASSERT_FALSE(samples.empty());
    for (const auto& s : samples) {
        EXPECT_TRUE(s.target == 1.0f || s.target == 0.0f || s.target == -1.0f);
        EXPECT_GT(std::popcount(s.planes.legal), 1);
    }
}

TEST(SelfPlay, BlendedTargetsMixResultAndSearch) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(1, net.heuristic());
    othello_bot::self_play::GameConfig config;

    // the same seed replays the same game: searching before a random move
    // doesn't touch the rng, and the player is deterministic
    auto play = [&](float lambda) {
        config.outcome_weight = lambda;
        std::mt19937_64 rng(5);
        return othello_bot::self_play::play_game(player, config, rng);
    };
    const auto result = play(1.0f);
    const auto search = play(0.0f);
    const auto mixed = play(0.5f);

    ASSERT_EQ(result.size(), search.size());
    ASSERT_EQ(result.size(), mixed.size());
    bool any_search_differs = false;
    for (std::size_t i = 0; i < result.size(); ++i) {
        ASSERT_EQ(result[i].planes.own, search[i].planes.own);
        EXPECT_GE(search[i].target, -1.0f);
        EXPECT_LE(search[i].target, 1.0f);
        EXPECT_FLOAT_EQ(mixed[i].target, 0.5f * result[i].target + 0.5f * search[i].target);
        any_search_differs |= search[i].target != result[i].target;
    }
    EXPECT_TRUE(any_search_differs);
}

TEST(SelfPlay, RejectsBadOutcomeWeight) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(1, net.heuristic());
    othello_bot::self_play::GameConfig config;
    config.outcome_weight = 1.5f;
    std::mt19937_64 rng(0);
    EXPECT_THROW(othello_bot::self_play::play_game(player, config, rng), std::invalid_argument);
}

TEST(SelfPlay, ReplayBufferDropsOldest) {
    othello_bot::self_play::ReplayBuffer buffer(2);
    const std::vector<Sample> samples = {{{}, 1.0f}, {{}, 0.0f}, {{}, -1.0f}};
    buffer.add(samples);
    EXPECT_EQ(buffer.size(), 2u);
    std::mt19937_64 rng(0);
    for (const auto& s : buffer.sample(20, rng)) {
        EXPECT_NE(s.target, 1.0f);
    }
}
