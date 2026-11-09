// Plays non-human players against each other and writes one CSV row per game.
//
//   ./build/benchmark CONFIG OUTPUT_CSV
//
// CONFIG is a text file of "key = value" settings and "player" lines ('#'
// starts a comment):
//
//   trials = 10          games per colour per matchup: each trial is one random
//                        opening played twice, once with each side as black
//   opening_plies = 4    random moves played before the players take over, so
//                        deterministic players don't replay the same game
//                        (0 starts every game from the initial position)
//   threads = 1          games played at once; 0 uses every hardware thread.
//                        More threads finish sooner but make per-move times
//                        noisier, since the games compete for the CPU
//   seed = 0             picks the openings and seeds MCTS; same seed, same openings
//
//   player NAME = TYPE key=value key=value ...
//
//     mcts       iterations=10000  c=1.41421  (UCT exploration constant)
//     alphabeta  depth=6  heuristic=mobility|discs  time_limit_ms=0 (0: none)
//                table_bits=16
//     network    depth=4  model=models/latest.ckpt  time_limit_ms=0  table_bits=16
//                (alpha-beta with the trained value network as heuristic)
//
//   match = NAME_A NAME_B   play only the listed matchups (any number of lines).
//                           Without any match lines, every pair of players meets
//                           once (round robin).
//
// Every game starts from fresh players (empty transposition tables / search
// trees), so results don't depend on which thread played which game. MCTS
// seeds are derived from the config seed and the game, so runs repeat.
//
// CSV columns: game, matchup, trial, opening_plies, black, black_type,
// black_params, white, white_type, white_params, black_discs, white_discs,
// winner (black/white/draw), margin (black - white), black_moves, white_moves,
// black_avg_move_seconds, white_avg_move_seconds, black_max_move_seconds,
// white_max_move_seconds, total_plies (moves played, passes not counted), moves.
//
// moves lists every move of the game in order, the random opening included
// (the first opening_plies of them), as space separated squares like "d3",
// with "pass" where a side had no legal move. Move counts and times cover
// only the moves each player chose itself (forced moves included).

#include <algorithm>
#include <bit>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <othello/othello.hpp>

#include <othello_bot/othello_bot.hpp>

namespace {

using othello_bot::MCTSPlayer;
using othello_bot::MinimaxPlayer;
using othello_bot::NetworkPlayer;
using Board = othello::board::Board;
using Clock = std::chrono::steady_clock;

std::string_view trim(std::string_view s) {
    const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && space(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && space(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

std::vector<std::string_view> split_whitespace(std::string_view s) {
    std::vector<std::string_view> parts;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
            ++i;
        }
        std::size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t') {
            ++j;
        }
        if (j > i) {
            parts.push_back(s.substr(i, j - i));
        }
        i = j;
    }
    return parts;
}

template <class T>
T parse_number(std::string_view text, std::string_view what) {
    text = trim(text);
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        throw std::invalid_argument(std::format("{} must be a number, got '{}'", what, text));
    }
    return value;
}

struct PlayerSpec {
    std::string name;
    std::string type;
    // only the keys that apply to `type` are used
    std::size_t iterations = 10'000;
    double c = std::numbers::sqrt2;
    int depth = 6;
    std::string heuristic = "mobility";
    std::string model = "models/latest.ckpt";
    int time_limit_ms = 0;
    int table_bits = MinimaxPlayer::DEFAULT_TABLE_BITS;

    // the settings that apply to `type`, as written to the CSV
    std::string params() const {
        std::string out;
        if (type == "mcts") {
            out = std::format("iterations={} c={:.6g}", iterations, c);
        } else if (type == "alphabeta") {
            out = std::format("depth={} heuristic={}", depth, heuristic);
        } else {
            out = std::format("depth={} model={}", depth, model);
        }
        if (type != "mcts") {
            if (time_limit_ms > 0) {
                out += std::format(" time_limit_ms={}", time_limit_ms);
            }
            if (table_bits != MinimaxPlayer::DEFAULT_TABLE_BITS) {
                out += std::format(" table_bits={}", table_bits);
            }
        }
        return out;
    }
};

struct Config {
    int trials = 10;
    int opening_plies = 4;
    int threads = 1;
    uint64_t seed = 0;
    std::vector<PlayerSpec> players;
    // indices into players
    std::vector<std::pair<std::size_t, std::size_t>> matches;
};

PlayerSpec parse_player(std::string_view name, std::string_view definition) {
    PlayerSpec spec;
    spec.name = std::string(trim(name));
    if (spec.name.empty()) {
        throw std::invalid_argument("player needs a name: 'player NAME = TYPE ...'");
    }
    const auto parts = split_whitespace(definition);
    if (parts.empty()) {
        throw std::invalid_argument(std::format("player {} needs a type", spec.name));
    }
    spec.type = std::string(parts[0]);
    if (spec.type != "mcts" && spec.type != "alphabeta" && spec.type != "network") {
        throw std::invalid_argument(std::format(
            "player {}: unknown type '{}' (expected mcts, alphabeta or network)", spec.name, spec.type
        ));
    }
    const bool searches = spec.type != "mcts";
    for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::string_view part = parts[i];
        const std::size_t eq = part.find('=');
        if (eq == std::string_view::npos) {
            throw std::invalid_argument(std::format("player {}: expected key=value, got '{}'", spec.name, part));
        }
        const std::string_view key = part.substr(0, eq);
        const std::string_view value = part.substr(eq + 1);
        const auto bad_key = [&] {
            return std::invalid_argument(
                std::format("player {}: '{}' is not a setting of {}", spec.name, key, spec.type)
            );
        };
        if (key == "iterations") {
            if (spec.type != "mcts") throw bad_key();
            spec.iterations = parse_number<std::size_t>(value, "iterations");
        } else if (key == "c") {
            if (spec.type != "mcts") throw bad_key();
            spec.c = parse_number<double>(value, "c");
        } else if (key == "depth") {
            if (!searches) throw bad_key();
            spec.depth = parse_number<int>(value, "depth");
        } else if (key == "heuristic") {
            if (spec.type != "alphabeta") throw bad_key();
            spec.heuristic = std::string(value);
            if (spec.heuristic != "mobility" && spec.heuristic != "discs") {
                throw std::invalid_argument(std::format(
                    "player {}: heuristic must be mobility or discs", spec.name
                ));
            }
        } else if (key == "model") {
            if (spec.type != "network") throw bad_key();
            spec.model = std::string(value);
        } else if (key == "time_limit_ms") {
            if (!searches) throw bad_key();
            spec.time_limit_ms = parse_number<int>(value, "time_limit_ms");
        } else if (key == "table_bits") {
            if (!searches) throw bad_key();
            spec.table_bits = parse_number<int>(value, "table_bits");
        } else {
            throw bad_key();
        }
    }
    if (spec.iterations < 1 || spec.depth < 1 || spec.time_limit_ms < 0 ||
        spec.table_bits < 0 || spec.table_bits > 30) {
        throw std::invalid_argument(std::format(
            "player {}: iterations and depth must be positive, time_limit_ms not negative, "
            "table_bits in [0, 30]",
            spec.name
        ));
    }
    return spec;
}

Config load_config(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::invalid_argument("could not open config file " + path.string());
    }
    Config config;
    std::vector<std::pair<std::string, std::string>> match_names;
    std::string line;
    for (int number = 1; std::getline(file, line); ++number) {
        try {
            std::string_view text = line;
            if (const std::size_t hash = text.find('#'); hash != std::string_view::npos) {
                text = text.substr(0, hash);
            }
            text = trim(text);
            if (text.empty()) {
                continue;
            }
            const std::size_t eq = text.find('=');
            if (eq == std::string_view::npos) {
                throw std::invalid_argument("expected 'key = value'");
            }
            const std::string_view key = trim(text.substr(0, eq));
            const std::string_view value = trim(text.substr(eq + 1));

            if (key.starts_with("player ") || key.starts_with("player\t")) {
                config.players.push_back(parse_player(key.substr(7), value));
            } else if (key == "match") {
                const auto names = split_whitespace(value);
                if (names.size() != 2) {
                    throw std::invalid_argument("expected 'match = NAME_A NAME_B'");
                }
                match_names.emplace_back(std::string(names[0]), std::string(names[1]));
            } else if (key == "trials") {
                config.trials = parse_number<int>(value, "trials");
            } else if (key == "opening_plies") {
                config.opening_plies = parse_number<int>(value, "opening_plies");
            } else if (key == "threads") {
                config.threads = parse_number<int>(value, "threads");
            } else if (key == "seed") {
                config.seed = parse_number<uint64_t>(value, "seed");
            } else {
                throw std::invalid_argument(std::format("unknown setting '{}'", key));
            }
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::format("{}:{}: {}", path.string(), number, e.what()));
        }
    }

    if (config.trials < 1 || config.opening_plies < 0 || config.threads < 0) {
        throw std::invalid_argument("trials must be positive, opening_plies and threads not negative");
    }
    if (config.threads == 0) {
        config.threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < config.players.size(); ++i) {
        if (!index.emplace(config.players[i].name, i).second) {
            throw std::invalid_argument(std::format("player {} is defined twice", config.players[i].name));
        }
    }
    if (match_names.empty()) {
        if (config.players.size() < 2) {
            throw std::invalid_argument("define at least two players");
        }
        for (std::size_t a = 0; a < config.players.size(); ++a) {
            for (std::size_t b = a + 1; b < config.players.size(); ++b) {
                config.matches.emplace_back(a, b);
            }
        }
    } else {
        for (const auto& [a, b] : match_names) {
            for (const auto& name : {a, b}) {
                if (!index.contains(name)) {
                    throw std::invalid_argument(std::format("match names unknown player {}", name));
                }
            }
            config.matches.emplace_back(index.at(a), index.at(b));
        }
    }
    return config;
}

// a fresh player for one game
std::unique_ptr<othello::player::Player> make_player(const PlayerSpec& spec, uint64_t seed) {
    std::optional<MinimaxPlayer::Clock::duration> limit;
    if (spec.time_limit_ms > 0) {
        limit = std::chrono::milliseconds(spec.time_limit_ms);
    }
    if (spec.type == "mcts") {
        return std::make_unique<MCTSPlayer>(spec.iterations, seed, spec.c);
    }
    if (spec.type == "alphabeta") {
        auto player = std::make_unique<MinimaxPlayer>(
            spec.depth,
            spec.heuristic == "discs" ? MinimaxPlayer::disc_difference : MinimaxPlayer::mobility,
            spec.table_bits
        );
        player->set_time_limit(limit);
        return player;
    }
    auto player = NetworkPlayer::load(spec.model, spec.depth, spec.table_bits);
    player->search().set_time_limit(limit);
    return player;
}

uint64_t random_move(uint64_t moves, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(0, std::popcount(moves) - 1);
    for (int k = dist(rng); k > 0; --k) {
        moves &= moves - 1;
    }
    return moves & -moves;
}

struct Game {
    std::size_t id = 0;
    std::size_t match = 0;
    int trial = 0;
    // index into Config::players
    std::size_t black = 0;
    std::size_t white = 0;
    // the random opening, played before the players take over
    std::vector<uint64_t> opening;
};

struct SideStats {
    int moves = 0;
    double seconds = 0.0;
    double max_seconds = 0.0;
};

struct GameResult {
    Board board;
    SideStats black;
    SideStats white;
    std::vector<std::string> moves;
};

void play_move(Board& board, uint64_t move, std::vector<std::string>& moves) {
    board.make_move(move);
    moves.push_back(othello::util::move_to_string(move));
    if (board.last_move_caused_pass()) {
        moves.emplace_back("pass");
    }
}

GameResult play(const Game& game, const Config& config) {
    GameResult result;
    for (const uint64_t move : game.opening) {
        play_move(result.board, move, result.moves);
    }
    // distinct per game and per side, so MCTS players don't mirror each other
    const uint64_t seed = config.seed * 0x9E3779B97F4A7C15ULL + game.id * 2;
    auto black = make_player(config.players[game.black], othello::util::mix(seed));
    auto white = make_player(config.players[game.white], othello::util::mix(seed + 1));

    Board& board = result.board;
    while (!board.game_over()) {
        const bool black_to_move = board.turn() == othello::Color::BLACK;
        auto& player = black_to_move ? *black : *white;
        SideStats& stats = black_to_move ? result.black : result.white;
        const auto start = Clock::now();
        const uint64_t move = player.get_move(board);
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (!board.move_valid(move) || (move & board.legal_moves()) == 0) {
            throw std::logic_error(std::format(
                "{} played an illegal move", config.players[black_to_move ? game.black : game.white].name
            ));
        }
        ++stats.moves;
        stats.seconds += seconds;
        stats.max_seconds = std::max(stats.max_seconds, seconds);
        play_move(board, move, result.moves);
    }
    return result;
}

// quotes a field if it contains anything CSV treats specially
std::string csv_field(std::string_view s) {
    if (s.find_first_of(",\"\n\r") == std::string_view::npos) {
        return std::string(s);
    }
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') {
            out += '"';
        }
        out += c;
    }
    return out + '"';
}

std::string csv_row(const Game& game, const GameResult& r, const Config& config) {
    const PlayerSpec& b = config.players[game.black];
    const PlayerSpec& w = config.players[game.white];
    const PlayerSpec& ma = config.players[config.matches[game.match].first];
    const PlayerSpec& mb = config.players[config.matches[game.match].second];
    const int margin = r.board.margin();
    const char* winner = margin > 0 ? "black" : margin < 0 ? "white" : "draw";
    std::string moves;
    for (const auto& m : r.moves) {
        moves += (moves.empty() ? "" : " ") + m;
    }
    const auto avg = [](const SideStats& s) { return s.moves ? s.seconds / s.moves : 0.0; };
    return std::format(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n",
        game.id, csv_field(ma.name + " vs " + mb.name), game.trial, game.opening.size(),
        csv_field(b.name), b.type, csv_field(b.params()),
        csv_field(w.name), w.type, csv_field(w.params()),
        r.board.black_count(), r.board.white_count(), winner, margin,
        r.black.moves, r.white.moves, avg(r.black), avg(r.white),
        r.black.max_seconds, r.white.max_seconds,
        std::ranges::count_if(r.moves, [](const auto& m) { return m != "pass"; }), moves
    );
}

// running totals of one matchup, from the point of view of its first player
struct Tally {
    int a_wins = 0;
    int draws = 0;
    int b_wins = 0;
    int a_margin = 0;
    SideStats a;
    SideStats b;
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::println(std::cerr, "usage: {} CONFIG OUTPUT_CSV", argv[0]);
        return 2;
    }
    Config config;
    try {
        config = load_config(argv[1]);
        // load each network once up front so a bad model path fails before any game
        for (const auto& spec : config.players) {
            if (spec.type == "network") {
                NetworkPlayer::load(spec.model, spec.depth, spec.table_bits);
            }
        }
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        return 2;
    }

    try {
        // the same openings for every matchup, so matchups are comparable
        std::mt19937_64 rng(config.seed ^ 0x9E3779B97F4A7C15ULL);
        std::vector<std::vector<uint64_t>> openings;
        for (int t = 0; t < config.trials; ++t) {
            Board board;
            std::vector<uint64_t> moves;
            for (int ply = 0; ply < config.opening_plies && !board.game_over(); ++ply) {
                const uint64_t move = random_move(board.legal_moves(), rng);
                board.make_move(move);
                moves.push_back(move);
            }
            openings.push_back(std::move(moves));
        }

        std::vector<Game> games;
        for (std::size_t m = 0; m < config.matches.size(); ++m) {
            const auto [a, b] = config.matches[m];
            for (int t = 0; t < config.trials; ++t) {
                for (const bool a_black : {true, false}) {
                    games.push_back(Game{
                        .id = games.size(),
                        .match = m,
                        .trial = t,
                        .black = a_black ? a : b,
                        .white = a_black ? b : a,
                        .opening = openings[static_cast<std::size_t>(t)],
                    });
                }
            }
        }

        std::ofstream csv(argv[2], std::ios::binary | std::ios::trunc);
        if (!csv) {
            throw std::runtime_error(std::format("could not open {}", argv[2]));
        }
        csv << "game,matchup,trial,opening_plies,black,black_type,black_params,white,white_type,"
               "white_params,black_discs,white_discs,winner,margin,black_moves,white_moves,"
               "black_avg_move_seconds,white_avg_move_seconds,black_max_move_seconds,"
               "white_max_move_seconds,total_plies,moves\n"
            << std::flush;

        std::println(
            "{} players, {} matchups, {} games ({} trials x 2 colours each), "
            "{} opening plies, {} thread{}",
            config.players.size(), config.matches.size(), games.size(), config.trials,
            config.opening_plies, config.threads, config.threads == 1 ? "" : "s"
        );

        std::vector<Tally> tallies(config.matches.size());
        std::atomic<std::size_t> next{0};
        std::mutex mutex;
        std::size_t done = 0;
        std::exception_ptr error;
        const auto start = Clock::now();

        auto work = [&] {
            try {
                for (std::size_t i; (i = next.fetch_add(1)) < games.size();) {
                    const Game& game = games[i];
                    const GameResult r = play(game, config);
                    const std::string row = csv_row(game, r, config);

                    std::lock_guard lock(mutex);
                    csv << row << std::flush;
                    Tally& t = tallies[game.match];
                    const bool a_black = game.black == config.matches[game.match].first;
                    const int a_margin = a_black ? r.board.margin() : -r.board.margin();
                    (a_margin > 0 ? t.a_wins : a_margin < 0 ? t.b_wins : t.draws)++;
                    t.a_margin += a_margin;
                    for (const auto& [side, stats] : {std::pair{&t.a, a_black ? r.black : r.white},
                                                      std::pair{&t.b, a_black ? r.white : r.black}}) {
                        side->moves += stats.moves;
                        side->seconds += stats.seconds;
                        side->max_seconds = std::max(side->max_seconds, stats.max_seconds);
                    }
                    ++done;
                    std::println(
                        "[{}/{}] {} (black) {}-{} {} (white)",
                        done, games.size(), config.players[game.black].name,
                        r.board.black_count(), r.board.white_count(), config.players[game.white].name
                    );
                    // stdout is fully buffered when redirected to a file; keep a log current
                    std::fflush(stdout);
                }
            } catch (...) {
                std::lock_guard lock(mutex);
                if (!error) {
                    error = std::current_exception();
                }
                next = games.size();
            }
        };
        {
            const int workers = std::clamp(config.threads, 1, static_cast<int>(games.size()));
            std::vector<std::jthread> pool;
            for (int t = 1; t < workers; ++t) {
                pool.emplace_back(work);
            }
            work();
        }
        if (error) {
            std::rethrow_exception(error);
        }

        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        std::println("\nsummary ({:.0f}s), first player's wins-draws-losses:", wall);
        for (std::size_t m = 0; m < config.matches.size(); ++m) {
            const auto& t = tallies[m];
            const auto& a = config.players[config.matches[m].first];
            const auto& b = config.players[config.matches[m].second];
            const int n = t.a_wins + t.draws + t.b_wins;
            const auto avg = [](const SideStats& s) { return s.moves ? s.seconds / s.moves : 0.0; };
            std::println(
                "  {} vs {}: {}-{}-{} ({:.1f}%), mean margin {:+.1f}, {:.4f}s/move vs {:.4f}s/move",
                a.name, b.name, t.a_wins, t.draws, t.b_wins, 100.0 * (t.a_wins + 0.5 * t.draws) / n,
                static_cast<double>(t.a_margin) / n, avg(t.a), avg(t.b)
            );
        }
        std::println("wrote {}", argv[2]);
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        return 1;
    }
    return 0;
}
