// Self-play training loop for the CNN value network used by MinimaxPlayer.
//
//   ./build/train                      start from a fresh network
//   ./build/train models/latest.mlfo   resume from a saved network
//
// Each iteration:
//   1. generate: the current network plays games against itself through
//      MinimaxPlayer, and every position is labelled with a blend of the
//      game's result and the search's value for it (see GameConfig)
//   2. train: Adam on random batches from a replay buffer of recent positions
//   3. evaluate (every few iterations): play the previous best network and a
//      fixed baseline; keep the new network as best if it wins often enough
//   4. checkpoint

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <print>
#include <random>

#include <othello_bot/othello_bot.hpp>

namespace {

using othello_bot::MinimaxPlayer;
using othello_bot::ValueNetwork;
namespace self_play = othello_bot::self_play;

// TODO: read these from the command line or a config file
struct Config {
    uint64_t seed = 0;
    std::size_t iterations = 100;

    // generation
    std::size_t games_per_iteration = 32;
    // keep this small: every leaf is a forward pass of the network
    int self_play_depth = 2;
    self_play::GameConfig game{};
    // outcome_weight (lambda) moves linearly from start to end over this many
    // iterations, then stays at end: lean on game results while the network
    // is still random, then on its (by then better) search values
    float outcome_weight_start = 1.0f;
    float outcome_weight_end = 0.5f;
    std::size_t outcome_weight_iterations = 50;

    // training
    std::size_t buffer_capacity = 200'000;
    std::size_t batch_size = 64;
    std::size_t train_steps_per_iteration = 200;
    float learning_rate = ValueNetwork::DEFAULT_LEARNING_RATE;

    // evaluation
    std::size_t evaluate_every = 5;
    int eval_games = 20;
    int eval_depth = 2;
    int eval_random_opening_plies = 4;
    // fraction of points against the best network needed to replace it
    double promote_threshold = 0.55;

    std::filesystem::path model_dir = "models";
};

float outcome_weight(const Config& config, std::size_t iteration) {
    if (config.outcome_weight_iterations <= 1) {
        return config.outcome_weight_end;
    }
    const double t = std::min(
        1.0,
        static_cast<double>(iteration - 1) / static_cast<double>(config.outcome_weight_iterations - 1)
    );
    return static_cast<float>(
        config.outcome_weight_start + t * (config.outcome_weight_end - config.outcome_weight_start)
    );
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char** argv) {
    const Config config;
    std::mt19937_64 rng(config.seed);
    std::filesystem::create_directories(config.model_dir);
    const auto latest_path = config.model_dir / "latest.mlfo";
    const auto best_path = config.model_dir / "best.mlfo";

    std::optional<ValueNetwork> loaded;
    try {
        loaded = (argc > 1)
            ? ValueNetwork::load(argv[1], config.batch_size, config.learning_rate)
            : ValueNetwork::create(static_cast<uint32_t>(config.seed), config.batch_size, config.learning_rate);
    } catch (const std::exception& e) {
        std::println(std::cerr, "{}", e.what());
        return 1;
    }
    ValueNetwork& net = *loaded;
    // the network the current one has to beat to be saved as best
    ValueNetwork best = net.clone();

    // players hold heuristics pointing at `net` / `best`, which stay put from here on
    MinimaxPlayer self_play_player(config.self_play_depth, net.heuristic());
    MinimaxPlayer candidate(config.eval_depth, net.heuristic());
    MinimaxPlayer champion(config.eval_depth, best.heuristic());
    MinimaxPlayer baseline(config.eval_depth, MinimaxPlayer::mobility);

    self_play::ReplayBuffer buffer(config.buffer_capacity);

    for (std::size_t iteration = 1; iteration <= config.iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();

        // 1. generate
        // TODO: generate games on several threads, one ValueNetwork clone per thread
        self_play::GameConfig game = config.game;
        game.outcome_weight = outcome_weight(config, iteration);
        std::size_t new_samples = 0;
        for (std::size_t g = 0; g < config.games_per_iteration; ++g) {
            const auto samples = self_play::play_game(self_play_player, game, rng);
            buffer.add(samples);
            new_samples += samples.size();
        }
        const double generate_seconds = seconds_since(start);

        // 2. train
        // TODO: learning rate schedule; L2 weight decay needs optimizer support in mlfo
        double loss = 0.0;
        for (std::size_t step = 0; step < config.train_steps_per_iteration; ++step) {
            const auto batch = buffer.sample(config.batch_size, rng);
            loss += net.train_batch(batch, rng);
        }
        loss /= static_cast<double>(config.train_steps_per_iteration);

        std::println(
            "iteration {}: lambda {:.2f}, {} new samples (buffer {}), generate {:.1f}s, total {:.1f}s, loss {:.5f}",
            iteration, game.outcome_weight, new_samples, buffer.size(), generate_seconds,
            seconds_since(start), loss
        );

        // 3. evaluate
        if (iteration % config.evaluate_every == 0) {
            const auto vs_best = self_play::match(
                candidate, champion, config.eval_games, config.eval_random_opening_plies, rng
            );
            const auto vs_baseline = self_play::match(
                candidate, baseline, config.eval_games, config.eval_random_opening_plies, rng
            );
            std::println(
                "  vs best {}-{}-{} ({:.0f}%), vs mobility {}-{}-{} ({:.0f}%)",
                vs_best.wins, vs_best.draws, vs_best.losses, 100 * vs_best.score(),
                vs_baseline.wins, vs_baseline.draws, vs_baseline.losses, 100 * vs_baseline.score()
            );
            if (vs_best.score() >= config.promote_threshold) {
                // in place, so champion's heuristic (which points at `best`) stays valid
                best.copy_weights_from(net);
                best.save(best_path);
                std::println("  promoted to best, saved {}", best_path.string());
            }
            // TODO: decide whether self-play should use `best` (AlphaGo Zero
            // style) or always the latest network (AlphaZero style, as now)
        }

        // 4. checkpoint
        net.save(latest_path);
    }
    return 0;
}
