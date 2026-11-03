// Self-play training loop for the CNN value network used by MinimaxPlayer.
//
//   ./build/train [--config FILE] [--key=value | --key value]...
//   ./build/train --resume RUN_DIR [--key=value | --key value]...
//
// e.g. ./build/train --iterations 20 --self_play_depth 3 --run_dir runs/test
// Arguments apply in order, so later ones override earlier ones. See
// TrainConfig (include/othello_bot/train_config.hpp) for every setting.
//
// Each iteration:
//   1. generate: the latest network plays games against itself through
//      MinimaxPlayer, and every position is labelled with a blend of the
//      game's result and the search's value for it (outcome_weight)
//   2. train: Adam on random batches from a replay buffer of recent positions,
//      at the learning rate the schedule gives this iteration (lr_schedule)
//   3. evaluate (every evaluate_every iterations, and after the last one):
//      play mobility minimax at each baseline depth, from the same paired
//      openings every time, so scores are comparable across the run
//   4. checkpoint
//
// Everything a run writes goes to run_dir:
//   config.txt     the full config; pass it back with --config to repeat the run
//   metrics.csv    one row per iteration
//   eval.csv       one row per evaluation match
//   latest.ckpt    the network and its optimizer state after the most recent iteration
//   checkpoints/   the same, every checkpoint_every iterations
//   resume.state   everything --resume needs: the network and optimizer, the
//                  replay buffer, the rng and the iteration count
// Files are written atomically, so an interrupted run never leaves a
// half-written one behind.
//
// Resuming: --resume RUN_DIR continues the run in RUN_DIR from its last
// completed iteration, with the settings in its config.txt. Any further
// arguments override those, e.g. --iterations 200 to extend a finished run.
// A resumed run plays the same games and reaches the same weights as one that
// was never interrupted, as long as the settings are not changed. Rows that an
// interrupted iteration had already added to the CSVs are dropped.
//
// --init_model starts a new run from a checkpoint (.ckpt), which restores the
// weights and Adam's state, or from a weights file (.mlfo), which starts a
// fresh optimizer. Either way, the replay buffer and iteration count start over.

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <othello_bot/othello_bot.hpp>

namespace {

using othello_bot::MinimaxPlayer;
using othello_bot::TrainConfig;
using othello_bot::ValueNetwork;
namespace self_play = othello_bot::self_play;

using Clock = std::chrono::steady_clock;

constexpr std::string_view METRICS_HEADER =
    "iteration,games,new_samples,buffer_size,generate_seconds,train_seconds,loss,"
    "network_evaluations,learning_rate";
constexpr std::string_view EVAL_HEADER = "iteration,opponent,games,wins,draws,losses,score";
constexpr std::string_view RUN_STATE_FILE = "resume.state";

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_usage(const char* program) {
    std::println(std::cerr, "usage: {} [--config FILE] [--key=value | --key value]...", program);
    std::println(std::cerr, "       {} --resume RUN_DIR [--key=value | --key value]...", program);
    std::println(std::cerr, "settings and their defaults:");
    std::print(std::cerr, "{}", TrainConfig{}.to_string());
}

std::filesystem::path default_run_dir() {
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    return std::filesystem::path("runs") / std::format("{:%Y%m%d-%H%M%S}", now);
}

void write_config(const TrainConfig& config) {
    othello_bot::write_atomically(config.run_dir / "config.txt", [&](const std::filesystem::path& tmp) {
        std::ofstream file(tmp);
        file << config.to_string();
        if (!file) {
            throw std::runtime_error("could not write " + tmp.string());
        }
    });
}

// a CSV file whose rows are flushed as they are written, so a crashed or
// interrupted run keeps everything up to that point
class CSV {
private:
    std::ofstream file_;

public:
    // a new file holding just the header
    static CSV create(const std::filesystem::path& path, std::string_view header) {
        CSV csv(path, std::ios::trunc);
        csv.row(header);
        return csv;
    }

    // an existing file of a resumed run: rows past last_iteration (the first
    // column) came from an iteration that did not finish, and are dropped
    static CSV resume(
        const std::filesystem::path& path, std::string_view header, std::size_t last_iteration
    ) {
        std::ifstream in(path);
        std::string line;
        if (!in || !std::getline(in, line) || line != header) {
            throw std::runtime_error(
                path.string() + " is missing or has different columns than this version writes"
            );
        }
        std::string kept = line + '\n';
        while (std::getline(in, line)) {
            std::size_t iteration = 0;
            std::istringstream fields(line);
            if (fields >> iteration && iteration <= last_iteration) {
                kept += line + '\n';
            }
        }
        in.close();
        othello_bot::write_atomically(path, [&](const std::filesystem::path& tmp) {
            std::ofstream out(tmp, std::ios::binary);
            out << kept;
            if (!out) {
                throw std::runtime_error("could not write " + tmp.string());
            }
        });
        return CSV(path, std::ios::app);
    }

    void row(std::string_view line) {
        file_ << line << '\n' << std::flush;
    }

private:
    CSV(const std::filesystem::path& path, std::ios::openmode mode) :
    file_(path, std::ios::out | std::ios::binary | mode)
    {
        if (!file_) {
            throw std::runtime_error("could not open " + path.string());
        }
    }
};

int run(TrainConfig config, bool resume) {
    if (config.run_dir.empty()) {
        config.run_dir = default_run_dir();
    }
    const auto checkpoint_dir = config.run_dir / "checkpoints";
    const auto state_path = config.run_dir / RUN_STATE_FILE;
    const std::string extension(ValueNetwork::CHECKPOINT_EXTENSION);

    std::optional<othello_bot::RunState> state;
    if (resume) {
        if (!std::filesystem::exists(state_path)) {
            throw std::runtime_error(
                state_path.string() + " not found: the run has no completed iteration to resume from"
            );
        }
        state = othello_bot::load_run_state(
            state_path, config.buffer_capacity, config.batch_size, config.learning_rate
        );
        if (state->iteration >= config.iterations) {
            std::println(
                "run_dir {} already finished {} of {} iterations; pass --iterations to extend it",
                config.run_dir.string(), state->iteration, config.iterations
            );
            return 0;
        }
    } else if (std::filesystem::exists(config.run_dir) && !std::filesystem::is_empty(config.run_dir)) {
        throw std::runtime_error(
            "run_dir " + config.run_dir.string() +
            " already exists and is not empty; use --resume to continue it"
        );
    }
    std::filesystem::create_directories(checkpoint_dir);
    // on resume this records any settings overridden on the command line
    write_config(config);

    const std::size_t first_iteration = state ? state->iteration + 1 : 1;
    CSV metrics = state
        ? CSV::resume(config.run_dir / "metrics.csv", METRICS_HEADER, state->iteration)
        : CSV::create(config.run_dir / "metrics.csv", METRICS_HEADER);
    CSV evals = state
        ? CSV::resume(config.run_dir / "eval.csv", EVAL_HEADER, state->iteration)
        : CSV::create(config.run_dir / "eval.csv", EVAL_HEADER);

    ValueNetwork net = state ? std::move(state->network)
        : config.init_model.empty()
        ? ValueNetwork::create(static_cast<uint32_t>(config.seed), config.batch_size, config.learning_rate)
        : ValueNetwork::load_any(config.init_model, config.batch_size, config.learning_rate);
    std::mt19937_64 rng = state ? state->rng : std::mt19937_64(config.seed);
    self_play::ReplayBuffer buffer = state
        ? std::move(state->buffer)
        : self_play::ReplayBuffer(config.buffer_capacity);
    state.reset();

    if (first_iteration == 1) {
        std::println("run_dir {}", config.run_dir.string());
    } else {
        std::println(
            "run_dir {}: resuming after iteration {} (buffer {})",
            config.run_dir.string(), first_iteration - 1, buffer.size()
        );
    }

    // the players hold heuristics pointing at `net`, which stays put from here on
    MinimaxPlayer self_play_player(config.self_play_depth, net.heuristic(), config.table_bits);
    MinimaxPlayer candidate(config.eval_depth, net.heuristic(), config.table_bits);
    std::vector<MinimaxPlayer> baselines;
    baselines.reserve(config.baseline_depths.size());
    for (const int depth : config.baseline_depths) {
        baselines.emplace_back(depth, MinimaxPlayer::mobility, config.table_bits);
    }

    for (std::size_t iteration = first_iteration; iteration <= config.iterations; ++iteration) {
        // Transposition tables keep best-move hints from earlier searches, which
        // can break ties between equal moves differently. Starting every
        // iteration from empty tables keeps an iteration's games a function of
        // what resume.state holds, so a resumed run matches an uninterrupted one.
        self_play_player.clear_table();
        candidate.clear_table();
        for (auto& baseline : baselines) {
            baseline.clear_table();
        }

        // 1. generate
        const auto generate_start = Clock::now();
        const std::size_t evaluations_before = net.evaluations();
        std::size_t new_samples = 0;
        for (std::size_t g = 0; g < config.games_per_iteration; ++g) {
            const auto samples = self_play::play_game(self_play_player, config.game, rng);
            buffer.add(samples);
            new_samples += samples.size();
        }
        const std::size_t evaluations = net.evaluations() - evaluations_before;
        const double generate_seconds = seconds_since(generate_start);

        // 2. train
        const float learning_rate = config.learning_rate_at(iteration);
        net.set_learning_rate(learning_rate);
        const auto train_start = Clock::now();
        double loss = 0.0;
        for (std::size_t step = 0; step < config.train_steps_per_iteration; ++step) {
            const auto batch = buffer.sample(config.batch_size, rng);
            loss += net.train_batch(batch, rng);
        }
        loss /= static_cast<double>(config.train_steps_per_iteration);
        const double train_seconds = seconds_since(train_start);

        std::println(
            "iteration {}: {} samples (buffer {}), generate {:.1f}s ({} evals), "
            "train {:.1f}s at lr {:.3g}, loss {:.5f}",
            iteration, new_samples, buffer.size(), generate_seconds, evaluations,
            train_seconds, learning_rate, loss
        );
        metrics.row(std::format(
            "{},{},{},{},{:.3f},{:.3f},{:.6f},{},{}",
            iteration, config.games_per_iteration, new_samples, buffer.size(),
            generate_seconds, train_seconds, loss, evaluations, learning_rate
        ));

        // 3. evaluate
        const bool last = iteration == config.iterations;
        if ((iteration % config.evaluate_every == 0 || last) && config.eval_openings > 0) {
            for (std::size_t i = 0; i < baselines.size(); ++i) {
                // same seed every evaluation: the same openings each time
                std::mt19937_64 eval_rng(config.seed ^ 0x9e3779b97f4a7c15ULL);
                const auto start = Clock::now();
                const auto r = self_play::match(
                    candidate, baselines[i], config.eval_openings,
                    config.eval_random_opening_plies, eval_rng
                );
                const std::string opponent = std::format("mobility_d{}", config.baseline_depths[i]);
                const int games = r.wins + r.draws + r.losses;
                std::println(
                    "  vs {}: {}-{}-{} ({:.1f}%) in {:.1f}s",
                    opponent, r.wins, r.draws, r.losses, 100 * r.score(), seconds_since(start)
                );
                evals.row(std::format(
                    "{},{},{},{},{},{},{:.4f}",
                    iteration, opponent, games, r.wins, r.draws, r.losses, r.score()
                ));
            }
        }

        // 4. checkpoint. The run state goes last: once it names this
        // iteration, every file above already holds this iteration's output.
        net.save_checkpoint(config.run_dir / ("latest" + extension));
        if (iteration % config.checkpoint_every == 0 || last) {
            net.save_checkpoint(checkpoint_dir / std::format("iter_{:06}{}", iteration, extension));
        }
        othello_bot::save_run_state(state_path, iteration, rng, buffer, net);
    }
    return 0;
}

// splits `--resume DIR` / `--resume=DIR` out of the arguments; the rest are
// left for TrainConfig::parse_args
std::optional<std::filesystem::path> take_resume(std::vector<const char*>& args) {
    std::optional<std::filesystem::path> dir;
    std::vector<const char*> rest;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--resume" || arg.starts_with("--resume=")) {
            if (dir) {
                throw std::invalid_argument("--resume given more than once");
            }
            if (arg == "--resume") {
                if (i + 1 >= args.size()) {
                    throw std::invalid_argument("--resume needs a value");
                }
                dir = args[++i];
            } else {
                dir = std::string(arg.substr(std::string_view("--resume=").size()));
            }
        } else {
            rest.push_back(args[i]);
        }
    }
    args = std::move(rest);
    return dir;
}

} // namespace

int main(int argc, char** argv) {
    TrainConfig config;
    std::optional<std::filesystem::path> resume;
    try {
        std::vector<const char*> args(argv + 1, argv + argc);
        resume = take_resume(args);
        if (resume) {
            config.load_file(*resume / "config.txt");
        }
        config.parse_args(static_cast<int>(args.size()), args.data());
        if (resume) {
            // the run lives where it was found, whatever its config says
            config.run_dir = *resume;
        }
        config.validate();
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        print_usage(argv[0]);
        return 2;
    }
    try {
        return run(std::move(config), resume.has_value());
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        return 1;
    }
}
