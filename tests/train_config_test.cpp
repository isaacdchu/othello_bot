#include <bit>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

using othello_bot::TrainConfig;

namespace {

std::filesystem::path temp_file(const std::string& name, const std::string& contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << contents;
    return path;
}

// always plays its lowest legal move: deterministic and stateless
class FirstMovePlayer : public othello::player::Player {
public:
    uint64_t get_move(const othello::board::Board& board) override {
        const uint64_t moves = board.legal_moves();
        return moves & -moves;
    }
};

// always plays its highest legal move: deterministic and stateless
class LastMovePlayer : public othello::player::Player {
public:
    uint64_t get_move(const othello::board::Board& board) override {
        return uint64_t{1} << (63 - std::countl_zero(board.legal_moves()));
    }
};

} // namespace

TEST(TrainConfig, DefaultsAreValid) {
    EXPECT_NO_THROW(TrainConfig{}.validate());
}

TEST(TrainConfig, SetsEveryKind) {
    TrainConfig c;
    c.set("seed", "12345678901234");
    c.set("self_play_depth", " 3 ");
    c.set("epsilon", "0.25");
    c.set("outcome_weight", "0.5");
    c.set("baseline_depths", "2,4,6");
    c.set("run_dir", "runs/x");
    EXPECT_EQ(c.seed, 12345678901234u);
    EXPECT_EQ(c.self_play_depth, 3);
    EXPECT_EQ(c.game.epsilon, 0.25);
    EXPECT_EQ(c.game.outcome_weight, 0.5f);
    EXPECT_EQ(c.baseline_depths, (std::vector<int>{2, 4, 6}));
    EXPECT_EQ(c.run_dir, std::filesystem::path("runs/x"));
    c.set("baseline_depths", "");
    EXPECT_TRUE(c.baseline_depths.empty());
}

TEST(TrainConfig, RejectsBadInput) {
    TrainConfig c;
    EXPECT_THROW(c.set("no_such_key", "1"), std::invalid_argument);
    EXPECT_THROW(c.set("iterations", "ten"), std::invalid_argument);
    EXPECT_THROW(c.set("iterations", "10x"), std::invalid_argument);
    EXPECT_THROW(c.set("iterations", "-1"), std::invalid_argument);
    EXPECT_THROW(c.set("baseline_depths", "2,,4"), std::invalid_argument);
    c.set("outcome_weight", "1.5");
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(TrainConfig, ToStringRoundTrips) {
    TrainConfig c;
    c.seed = 99;
    c.learning_rate = 3e-4f;
    c.game.epsilon = 0.1;
    c.baseline_depths = {1, 3};
    c.run_dir = "runs/a b";
    const auto path = temp_file("othello_bot_config_roundtrip.txt", c.to_string());
    TrainConfig read;
    read.load_file(path);
    EXPECT_EQ(read.to_string(), c.to_string());
    std::filesystem::remove(path);
}

TEST(TrainConfig, ScheduleSettingsRoundTrip) {
    TrainConfig c;
    c.set("lr_schedule", " cosine ");
    c.set("lr_warmup_iterations", "3");
    c.set("lr_decay", "0.25");
    c.set("lr_decay_every", "7");
    c.set("lr_min", "2e-6");
    EXPECT_EQ(c.lr_schedule, "cosine");
    const auto path = temp_file("othello_bot_config_schedule.txt", c.to_string());
    TrainConfig read;
    read.load_file(path);
    EXPECT_EQ(read.to_string(), c.to_string());
    EXPECT_NO_THROW(read.validate());
    std::filesystem::remove(path);
}

TEST(TrainConfig, RejectsBadSchedule) {
    auto invalid = [](std::string_view key, std::string_view value) {
        TrainConfig c;
        c.set(key, value);
        EXPECT_THROW(c.validate(), std::invalid_argument) << key << " = " << value;
    };
    invalid("lr_schedule", "linear");
    invalid("lr_decay", "0");
    invalid("lr_decay", "1.5");
    invalid("lr_decay_every", "0");
    invalid("lr_min", "0");
    invalid("lr_min", "1");  // above the default learning_rate
}

TEST(TrainConfig, ConstantSchedule) {
    TrainConfig c;
    c.learning_rate = 0.01f;
    for (std::size_t i = 1; i <= c.iterations; ++i) {
        EXPECT_EQ(c.learning_rate_at(i), 0.01f);
    }
}

TEST(TrainConfig, StepSchedule) {
    TrainConfig c;
    c.learning_rate = 0.01f;
    c.lr_schedule = "step";
    c.lr_decay = 0.5f;
    c.lr_decay_every = 10;
    c.lr_min = 1e-3f;
    EXPECT_FLOAT_EQ(c.learning_rate_at(1), 0.01f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(10), 0.01f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(11), 0.005f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(21), 0.0025f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(31), 0.00125f);
    // 0.000625 would be below lr_min
    EXPECT_FLOAT_EQ(c.learning_rate_at(41), 1e-3f);
}

TEST(TrainConfig, CosineSchedule) {
    TrainConfig c;
    c.iterations = 101;
    c.learning_rate = 0.01f;
    c.lr_schedule = "cosine";
    c.lr_min = 1e-4f;
    EXPECT_FLOAT_EQ(c.learning_rate_at(1), 0.01f);
    // halfway through, half way between the two
    EXPECT_NEAR(c.learning_rate_at(51), (0.01f + 1e-4f) / 2, 1e-7);
    EXPECT_FLOAT_EQ(c.learning_rate_at(101), 1e-4f);
    // and stays there if run past the end
    EXPECT_FLOAT_EQ(c.learning_rate_at(150), 1e-4f);
    for (std::size_t i = 2; i <= c.iterations; ++i) {
        EXPECT_LE(c.learning_rate_at(i), c.learning_rate_at(i - 1));
    }
}

TEST(TrainConfig, WarmupThenSchedule) {
    TrainConfig c;
    c.iterations = 14;
    c.learning_rate = 0.01f;
    c.lr_warmup_iterations = 4;
    c.lr_schedule = "cosine";
    c.lr_min = 1e-4f;
    EXPECT_FLOAT_EQ(c.learning_rate_at(1), 0.0025f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(2), 0.005f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(4), 0.01f);
    // the cosine starts after the warmup and still ends on the last iteration
    EXPECT_FLOAT_EQ(c.learning_rate_at(5), 0.01f);
    EXPECT_FLOAT_EQ(c.learning_rate_at(14), 1e-4f);
}

TEST(TrainConfig, ScheduleIsAlwaysPositive) {
    for (const char* schedule : {"constant", "step", "cosine"}) {
        TrainConfig c;
        c.lr_schedule = schedule;
        c.lr_warmup_iterations = 5;
        c.lr_decay = 0.1f;
        c.lr_decay_every = 1;
        c.validate();
        for (std::size_t i = 0; i <= c.iterations + 10; ++i) {
            EXPECT_GT(c.learning_rate_at(i), 0.0f) << schedule << " at " << i;
        }
    }
}

TEST(TrainConfig, ArgsApplyInOrder) {
    const auto path = temp_file(
        "othello_bot_config_args.txt",
        "# a comment\n\niterations = 7   # trailing comment\nbatch_size = 16\n"
    );
    const std::string config_path = path.string();
    const char* argv[] = {
        "--iterations=3", "--config", config_path.c_str(), "--batch_size", "32", "--seed=5",
    };
    TrainConfig c;
    c.parse_args(6, argv);
    EXPECT_EQ(c.iterations, 7u);   // the file came after --iterations=3
    EXPECT_EQ(c.batch_size, 32u);  // the argument came after the file
    EXPECT_EQ(c.seed, 5u);
    std::filesystem::remove(path);
}

TEST(TrainConfig, RejectsBadArgs) {
    TrainConfig c;
    const char* positional[] = {"iterations=3"};
    EXPECT_THROW(c.parse_args(1, positional), std::invalid_argument);
    const char* missing[] = {"--iterations"};
    EXPECT_THROW(c.parse_args(1, missing), std::invalid_argument);
    const char* no_file[] = {"--config", "does/not/exist.txt"};
    EXPECT_THROW(c.parse_args(2, no_file), std::invalid_argument);
}

TEST(Match, PairedOpeningsAreFair) {
    // identical deterministic players: each opening's two games mirror each
    // other, so every pair is one win and one loss, or two draws
    FirstMovePlayer a;
    FirstMovePlayer b;
    std::mt19937_64 rng(0);
    const auto r = othello_bot::self_play::match(a, b, 10, 4, rng);
    EXPECT_EQ(r.wins + r.draws + r.losses, 20);
    EXPECT_EQ(r.wins, r.losses);
    EXPECT_DOUBLE_EQ(r.score(), 0.5);
}

TEST(Match, SameSeedSameOpenings) {
    othello_bot::MinimaxPlayer strong(3, othello_bot::MinimaxPlayer::mobility);
    othello::player::RandomPlayer weak(1);
    std::mt19937_64 rng1(42);
    std::mt19937_64 rng2(42);
    FirstMovePlayer first_a;
    FirstMovePlayer first_b;
    // with deterministic players, equal seeds give equal results
    const auto r1 = othello_bot::self_play::match(first_a, first_b, 5, 6, rng1);
    const auto r2 = othello_bot::self_play::match(first_a, first_b, 5, 6, rng2);
    EXPECT_EQ(r1.wins, r2.wins);
    EXPECT_EQ(r1.draws, r2.draws);
    // and a real search beats random from the same openings
    std::mt19937_64 rng3(42);
    EXPECT_GT(othello_bot::self_play::match(strong, weak, 5, 6, rng3).score(), 0.5);
}

TEST(Match, ParallelMatchesSerial) {
    // deterministic, stateless players: every thread count plays the same games
    std::mt19937_64 rng(7);
    FirstMovePlayer first;
    LastMovePlayer last;
    const auto serial = othello_bot::self_play::match(first, last, 6, 4, rng);
    std::mt19937_64 rng2(7);
    const auto openings = othello_bot::self_play::random_openings(6, 4, rng2);
    for (const int threads : {1, 4}) {
        const auto r = othello_bot::self_play::parallel_match(
            [] { return std::make_unique<FirstMovePlayer>(); },
            [] { return std::make_unique<LastMovePlayer>(); },
            openings, threads
        );
        EXPECT_EQ(r.games(), 12);
        EXPECT_EQ(r.wins, serial.wins);
        EXPECT_EQ(r.draws, serial.draws);
        EXPECT_EQ(r.margin, serial.margin);
        EXPECT_GT(r.a_moves, 0);
        EXPECT_GT(r.b_moves, 0);
    }
}

TEST(TrainConfig, ThreadsAndMarginWeight) {
    TrainConfig c;
    c.set("threads", "4");
    c.set("margin_weight", "0.25");
    EXPECT_EQ(c.threads, 4u);
    EXPECT_EQ(c.game.margin_weight, 0.25f);
    c.set("margin_weight", "2");
    EXPECT_THROW(c.validate(), std::invalid_argument);
}
