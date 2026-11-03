#ifndef TRAIN_CONFIG_HPP
#define TRAIN_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "minimax_player.hpp"
#include "self_play.hpp"
#include "value_network.hpp"

namespace othello_bot {

// Settings of a training run. Every field can be set from a config file
// ("key = value" lines, '#' starts a comment) or the command line
// ("--key=value" or "--key value"), using the names from keys().
// to_string() writes the same format, so a run's saved config.txt can be
// passed back with --config to repeat it.
struct TrainConfig {
    uint64_t seed = 0;
    std::size_t iterations = 100;

    // generation
    std::size_t games_per_iteration = 32;
    // keep this small: every leaf is a forward pass of the network
    int self_play_depth = 2;
    // random_opening_plies, epsilon and outcome_weight (lambda, constant for the run)
    self_play::GameConfig game{};

    // training
    std::size_t buffer_capacity = 200'000;
    std::size_t batch_size = 64;
    std::size_t train_steps_per_iteration = 200;
    float learning_rate = ValueNetwork::DEFAULT_LEARNING_RATE;

    // learning rate schedule, applied once per iteration by learning_rate_at():
    //   constant  learning_rate throughout
    //   step      learning_rate * lr_decay^k, where k counts completed blocks of
    //             lr_decay_every iterations, never below lr_min
    //   cosine    learning_rate down to lr_min along a half cosine, reaching
    //             lr_min at the last iteration
    // with a linear warmup from learning_rate / lr_warmup_iterations up to
    // learning_rate over the first lr_warmup_iterations iterations; the
    // schedule proper starts after it
    std::string lr_schedule = "constant";
    std::size_t lr_warmup_iterations = 0;
    float lr_decay = 0.5f;
    std::size_t lr_decay_every = 50;
    float lr_min = 1e-5f;

    // evaluation: the network at eval_depth plays mobility minimax at each of
    // baseline_depths, eval_openings random openings each played with both colours
    std::size_t evaluate_every = 5;
    int eval_depth = 2;
    std::vector<int> baseline_depths = {1, 2, 4};
    int eval_openings = 20;
    int eval_random_opening_plies = 4;

    // search
    int table_bits = MinimaxPlayer::DEFAULT_TABLE_BITS;

    // output: empty run_dir means runs/<UTC timestamp>
    std::filesystem::path run_dir;
    std::size_t checkpoint_every = 10;
    // network to start from: a checkpoint (.ckpt) restores the optimizer state
    // too, a weights file (.mlfo) does not; empty starts a fresh network
    std::filesystem::path init_model;

    // every settable key, in the order to_string() writes them
    static std::vector<std::string_view> keys();

    // throws std::invalid_argument naming the key for an unknown key or a bad value
    void set(std::string_view key, std::string_view value);

    void load_file(const std::filesystem::path& path);

    // applies arguments in order: "--config FILE" loads a file, "--key=value"
    // and "--key value" set one key, so later arguments override earlier ones
    void parse_args(int argc, const char* const* argv);

    // throws std::invalid_argument describing the first bad setting
    void validate() const;

    // the learning rate for 1-based `iteration` under lr_schedule; always
    // positive for a valid config. Depends on `iterations` for cosine, so
    // extending a run with --iterations stretches the remaining decay.
    float learning_rate_at(std::size_t iteration) const;

    std::string to_string() const;
};

} // namespace othello_bot

#endif // TRAIN_CONFIG_HPP
