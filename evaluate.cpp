// Plays a trained network against mobility minimax at several depths, on many
// threads, and reports the score and thinking time of each match.
//
//   ./build/evaluate MODEL [--depth N] [--baselines 8,11] [--openings N]
//                          [--plies N] [--threads N] [--seed N] [--csv FILE]
//
//   MODEL        a checkpoint (.ckpt) or weights file (.mlfo) written by train
//   --depth      plies the network searches (default 4)
//   --baselines  comma separated mobility minimax depths (default 8,11)
//   --openings   random openings, each played with both colours (default 50)
//   --plies      random plies per opening (default 4)
//   --threads    worker threads (default: every hardware thread)
//   --seed       picks the openings; the same seed gives the same openings (default 0)
//   --csv        appends one row per baseline to FILE (written with a header if new)
//
// Pass "mobility_dN" as MODEL to evaluate mobility minimax at depth N instead,
// e.g. to see how two baseline depths compare.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <othello_bot/othello_bot.hpp>

namespace {

using othello_bot::MinimaxPlayer;
using othello_bot::NetworkPlayer;
namespace self_play = othello_bot::self_play;

int parse_int(std::string_view text, std::string_view what) {
    int value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        throw std::invalid_argument(std::format("{} must be a number, got '{}'", what, text));
    }
    return value;
}

std::vector<int> parse_list(std::string_view text) {
    std::vector<int> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        values.push_back(parse_int(text.substr(0, comma), "--baselines"));
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1);
    }
    return values;
}

struct Options {
    std::string model;
    int depth = 4;
    std::vector<int> baselines = {8, 11};
    int openings = 50;
    int plies = 4;
    int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    uint64_t seed = 0;
    std::filesystem::path csv;
};

Options parse(int argc, char** argv) {
    Options o;
    std::vector<std::string_view> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view arg = args[i];
        if (!arg.starts_with("--")) {
            if (!o.model.empty()) {
                throw std::invalid_argument(std::format("unexpected argument '{}'", arg));
            }
            o.model = std::string(arg);
            continue;
        }
        std::string_view key = arg.substr(2);
        std::string_view value;
        if (const auto eq = key.find('='); eq != std::string_view::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
        } else {
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(std::format("--{} needs a value", key));
            }
            value = args[++i];
        }
        if (key == "depth") {
            o.depth = parse_int(value, "--depth");
        } else if (key == "baselines") {
            o.baselines = parse_list(value);
        } else if (key == "openings") {
            o.openings = parse_int(value, "--openings");
        } else if (key == "plies") {
            o.plies = parse_int(value, "--plies");
        } else if (key == "threads") {
            o.threads = parse_int(value, "--threads");
        } else if (key == "seed") {
            o.seed = static_cast<uint64_t>(parse_int(value, "--seed"));
        } else if (key == "csv") {
            o.csv = std::string(value);
        } else {
            throw std::invalid_argument(std::format("unknown option --{}", key));
        }
    }
    if (o.model.empty()) {
        throw std::invalid_argument("MODEL is required");
    }
    if (o.depth < 1 || o.openings < 1 || o.plies < 0 || o.threads < 1 || o.baselines.empty()) {
        throw std::invalid_argument("depth, openings and threads must be positive, plies not negative");
    }
    for (const int d : o.baselines) {
        if (d < 1) {
            throw std::invalid_argument("baseline depths must be at least 1");
        }
    }
    return o;
}

// a factory for MODEL: a network player, or mobility minimax for "mobility_dN"
self_play::PlayerFactory candidate_factory(const Options& o) {
    constexpr std::string_view MOBILITY = "mobility_d";
    if (std::string_view(o.model).starts_with(MOBILITY)) {
        const int depth = parse_int(std::string_view(o.model).substr(MOBILITY.size()), "MODEL depth");
        return [depth] { return std::make_unique<MinimaxPlayer>(depth, MinimaxPlayer::mobility); };
    }
    // load once up front so a bad path fails before any thread starts
    NetworkPlayer::load(o.model, o.depth);
    return [path = o.model, depth = o.depth] { return NetworkPlayer::load(path, depth); };
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        std::println(
            std::cerr,
            "usage: {} MODEL [--depth N] [--baselines 8,11] [--openings N] [--plies N] "
            "[--threads N] [--seed N] [--csv FILE]",
            argv[0]
        );
        return 2;
    }
    try {
        const auto make_candidate = candidate_factory(o);
        std::mt19937_64 rng(o.seed ^ 0x9e3779b97f4a7c15ULL);
        const auto openings = self_play::random_openings(o.openings, o.plies, rng);
        std::println(
            "{} at depth {}: {} openings x 2 colours, {} random plies, {} threads",
            o.model, o.depth, o.openings, o.plies, o.threads
        );

        std::ofstream csv;
        if (!o.csv.empty()) {
            const bool fresh = !std::filesystem::exists(o.csv);
            csv.open(o.csv, std::ios::app | std::ios::binary);
            if (!csv) {
                throw std::runtime_error("could not open " + o.csv.string());
            }
            if (fresh) {
                csv << "model,depth,opponent,games,wins,draws,losses,score,mean_margin,"
                       "seconds_per_move,opponent_seconds_per_move\n";
            }
        }

        for (const int depth : o.baselines) {
            const auto start = std::chrono::steady_clock::now();
            const auto r = self_play::parallel_match(
                make_candidate,
                [depth] { return std::make_unique<MinimaxPlayer>(depth, MinimaxPlayer::mobility); },
                openings, o.threads
            );
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double own = r.a_moves ? r.a_seconds / r.a_moves : 0.0;
            const double theirs = r.b_moves ? r.b_seconds / r.b_moves : 0.0;
            const double margin = static_cast<double>(r.margin) / r.games();
            std::println(
                "  vs mobility_d{:<2}: {}-{}-{} ({:.1f}%), mean margin {:+.1f}, "
                "{:.3f}s/move vs {:.3f}s/move, {:.0f}s",
                depth, r.wins, r.draws, r.losses, 100 * r.score(), margin, own, theirs, wall
            );
            if (csv) {
                csv << std::format(
                    "{},{},mobility_d{},{},{},{},{},{:.4f},{:.2f},{:.4f},{:.4f}\n",
                    o.model, o.depth, depth, r.games(), r.wins, r.draws, r.losses, r.score(),
                    margin, own, theirs
                ) << std::flush;
            }
        }
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        return 1;
    }
    return 0;
}
