#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <set>
#include <sstream>
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
    EXPECT_EQ(x[(4 * 8 + 3) * enc::CHANNELS + 0], 1.0f);
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

TEST(Encoding, TransformMatchesBoardIsomorphism) {
    othello::board::Board board;
    othello::player::RandomPlayer random(3);
    for (int i = 0; i < 12; ++i) {
        board.make_move(random.get_move(board));
    }
    const auto p = enc::planes(board);
    for (int s = 1; s < enc::SYMMETRIES; ++s) {
        const auto symmetry = static_cast<othello::board::Board::Symmetry>(s - 1);
        // planes of the transformed board, legal moves recomputed from scratch
        const auto expected = enc::planes(othello::board::Board::isomorphism(board, symmetry));
        const auto t = enc::transform(p, s);
        EXPECT_EQ(t.own, expected.own) << "symmetry " << s;
        EXPECT_EQ(t.opp, expected.opp) << "symmetry " << s;
        EXPECT_EQ(t.legal, expected.legal) << "symmetry " << s;
    }
}

namespace {

// a fresh, empty directory under the system temp directory, removed on destruction
class TempDir {
private:
    std::filesystem::path path_;

public:
    explicit TempDir(std::string_view name) :
    path_(std::filesystem::temp_directory_path() / std::filesystem::path(std::string(name)))
    {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const {
        return path_;
    }
};

std::vector<Sample> random_batch(std::size_t size, int seed) {
    othello::player::RandomPlayer random(seed);
    std::vector<Sample> batch;
    othello::board::Board board;
    while (batch.size() < size && !board.game_over()) {
        batch.push_back({enc::planes(board), (batch.size() % 2 == 0) ? 1.0f : -1.0f});
        board.make_move(random.get_move(board));
    }
    return batch;
}

} // namespace

TEST(ValueNetwork, CheckpointResumesTrainingExactly) {
    TempDir dir("othello_bot_checkpoint_test");
    const auto batch = random_batch(8, 4);
    std::mt19937_64 rng(0);
    auto net = ValueNetwork::create(1, 8);
    for (int i = 0; i < 5; ++i) {
        net.train_batch(batch, rng);
    }
    const auto path = dir.path() / "net.ckpt";
    net.save_checkpoint(path);
    EXPECT_FALSE(std::filesystem::exists(dir.path() / "net.ckpt.tmp"));

    auto resumed = ValueNetwork::load_checkpoint(path, 8);
    // weights only: same starting point, but Adam starts over
    net.save(dir.path() / "net.mlfo");
    auto restarted = ValueNetwork::load(dir.path() / "net.mlfo", 8);

    const othello::board::Board board;
    ASSERT_EQ(net.evaluate(board), resumed.evaluate(board));
    ASSERT_EQ(net.evaluate(board), restarted.evaluate(board));

    // the same symmetries are drawn for all three, so only the optimizer state differs
    std::mt19937_64 rng_net(7);
    std::mt19937_64 rng_resumed(7);
    std::mt19937_64 rng_restarted(7);
    for (int i = 0; i < 3; ++i) {
        net.train_batch(batch, rng_net);
        resumed.train_batch(batch, rng_resumed);
        restarted.train_batch(batch, rng_restarted);
    }
    EXPECT_EQ(net.evaluate(board), resumed.evaluate(board));
    EXPECT_NE(net.evaluate(board), restarted.evaluate(board));
}

TEST(ValueNetwork, LoadAnyPicksFormatByExtension) {
    TempDir dir("othello_bot_load_any_test");
    auto net = ValueNetwork::create(2, 4);
    net.save(dir.path() / "net.mlfo");
    net.save_checkpoint(dir.path() / "net.ckpt");
    const othello::board::Board board;
    EXPECT_EQ(ValueNetwork::load_any(dir.path() / "net.mlfo", 4).evaluate(board), net.evaluate(board));
    EXPECT_EQ(ValueNetwork::load_any(dir.path() / "net.ckpt", 4).evaluate(board), net.evaluate(board));
    // a weights file is not a checkpoint
    EXPECT_ANY_THROW(ValueNetwork::load_checkpoint(dir.path() / "net.mlfo", 4));
}

TEST(ValueNetwork, SaveReplacesExistingFile) {
    TempDir dir("othello_bot_save_replace_test");
    const auto path = dir.path() / "net.ckpt";
    auto a = ValueNetwork::create(5, 4);
    auto b = ValueNetwork::create(6, 4);
    a.save_checkpoint(path);
    b.save_checkpoint(path);
    const othello::board::Board board;
    EXPECT_EQ(ValueNetwork::load_checkpoint(path, 4).evaluate(board), b.evaluate(board));
}

TEST(ValueNetwork, SetLearningRate) {
    auto net = ValueNetwork::create(0, 4, 0.01f);
    EXPECT_EQ(net.learning_rate(), 0.01f);
    net.set_learning_rate(0.002f);
    EXPECT_EQ(net.learning_rate(), 0.002f);
    EXPECT_THROW(net.set_learning_rate(0.0f), std::invalid_argument);
    EXPECT_EQ(net.learning_rate(), 0.002f);
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

TEST(SelfPlay, MarginWeightBlendsInDiscMargin) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(1, net.heuristic());
    othello_bot::self_play::GameConfig config;
    auto play = [&](float mu) {
        config.margin_weight = mu;
        std::mt19937_64 rng(5);
        return othello_bot::self_play::play_game(player, config, rng);
    };
    const auto wdl = play(0.0f);
    const auto margin = play(1.0f);
    const auto mixed = play(0.25f);
    ASSERT_EQ(wdl.size(), margin.size());
    ASSERT_EQ(wdl.size(), mixed.size());
    for (std::size_t i = 0; i < wdl.size(); ++i) {
        // a pure margin target is margin / 64, with the winner's sign
        EXPECT_LE(std::abs(margin[i].target), 1.0f);
        EXPECT_EQ(margin[i].target * 64.0f, std::round(margin[i].target * 64.0f));
        EXPECT_TRUE(wdl[i].target == 0.0f || (margin[i].target > 0) == (wdl[i].target > 0));
        EXPECT_FLOAT_EQ(mixed[i].target, 0.75f * wdl[i].target + 0.25f * margin[i].target);
    }
    config.margin_weight = -0.1f;
    std::mt19937_64 rng(0);
    EXPECT_THROW(othello_bot::self_play::play_game(player, config, rng), std::invalid_argument);
}

TEST(SelfPlay, RejectsBadOutcomeWeight) {
    auto net = ValueNetwork::create(0, 1);
    MinimaxPlayer player(1, net.heuristic());
    othello_bot::self_play::GameConfig config;
    config.outcome_weight = 1.5f;
    std::mt19937_64 rng(0);
    EXPECT_THROW(othello_bot::self_play::play_game(player, config, rng), std::invalid_argument);
}

TEST(SelfPlay, ReplayBufferRoundTrips) {
    othello_bot::self_play::ReplayBuffer buffer(10);
    const auto samples = random_batch(6, 8);
    buffer.add(samples);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    buffer.write(stream);

    const auto same = othello_bot::self_play::ReplayBuffer::read(stream, 10);
    ASSERT_EQ(same.size(), samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        EXPECT_EQ(same.samples()[i].planes.own, samples[i].planes.own);
        EXPECT_EQ(same.samples()[i].planes.opp, samples[i].planes.opp);
        EXPECT_EQ(same.samples()[i].planes.legal, samples[i].planes.legal);
        EXPECT_EQ(same.samples()[i].target, samples[i].target);
    }

    // a smaller capacity keeps the newest
    stream.clear();
    stream.seekg(0);
    const auto smaller = othello_bot::self_play::ReplayBuffer::read(stream, 4);
    ASSERT_EQ(smaller.size(), 4u);
    EXPECT_EQ(smaller.samples().front().planes.own, samples[2].planes.own);
    EXPECT_EQ(smaller.samples().back().planes.own, samples[5].planes.own);
}

TEST(SelfPlay, ReplayBufferRejectsTruncatedInput) {
    othello_bot::self_play::ReplayBuffer buffer(10);
    buffer.add(random_batch(3, 8));
    std::stringstream full(std::ios::in | std::ios::out | std::ios::binary);
    buffer.write(full);
    const std::string bytes = full.str();
    std::stringstream cut(bytes.substr(0, bytes.size() - 5), std::ios::in | std::ios::binary);
    EXPECT_THROW(othello_bot::self_play::ReplayBuffer::read(cut, 10), std::runtime_error);
    std::stringstream garbage("not a buffer\n", std::ios::in | std::ios::binary);
    EXPECT_THROW(othello_bot::self_play::ReplayBuffer::read(garbage, 10), std::runtime_error);
}

TEST(RunState, ResumedTrainingMatchesUninterrupted) {
    TempDir dir("othello_bot_run_state_test");
    const auto path = dir.path() / "resume.state";

    // one "iteration": train on samples drawn from the buffer with rng
    auto step = [](ValueNetwork& net, othello_bot::self_play::ReplayBuffer& buffer, std::mt19937_64& rng) {
        buffer.add(random_batch(8, static_cast<int>(rng() % 1000)));
        for (int i = 0; i < 3; ++i) {
            net.train_batch(buffer.sample(4, rng), rng);
        }
    };

    auto net = ValueNetwork::create(9, 4);
    othello_bot::self_play::ReplayBuffer buffer(100);
    std::mt19937_64 rng(11);
    step(net, buffer, rng);
    step(net, buffer, rng);
    othello_bot::save_run_state(path, 2, rng, buffer, net);
    EXPECT_FALSE(std::filesystem::exists(dir.path() / "resume.state.tmp"));

    auto state = othello_bot::load_run_state(path, 100, 4, ValueNetwork::DEFAULT_LEARNING_RATE);
    EXPECT_EQ(state.iteration, 2u);
    EXPECT_EQ(state.buffer.size(), buffer.size());

    step(net, buffer, rng);
    step(state.network, state.buffer, state.rng);
    const othello::board::Board board;
    EXPECT_EQ(state.network.evaluate(board), net.evaluate(board));
    EXPECT_EQ(state.rng(), rng());
}

TEST(RunState, RejectsOtherFiles) {
    TempDir dir("othello_bot_run_state_reject_test");
    auto net = ValueNetwork::create(0, 4);
    net.save_checkpoint(dir.path() / "net.ckpt");
    EXPECT_THROW(
        othello_bot::load_run_state(dir.path() / "net.ckpt", 10, 4, 1e-3f), std::runtime_error
    );
    EXPECT_THROW(
        othello_bot::load_run_state(dir.path() / "missing.state", 10, 4, 1e-3f), std::runtime_error
    );
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
