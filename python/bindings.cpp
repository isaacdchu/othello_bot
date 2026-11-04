// Python bindings for the board and the bots, used by the GUI (othello_gui).
//
// Squares are indices 0..63: row * 8 + col, so 0 is a1, 7 is h1 and 63 is h8,
// matching the board's string layout.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

namespace nb = nanobind;
using namespace nb::literals;

using othello::Color;
using othello::board::Board;
using othello::player::Player;
using othello_bot::MinimaxPlayer;

namespace {

uint64_t to_bit(int square) {
    if (square < 0 || square > 63) {
        throw nb::index_error("square must be in 0..63");
    }
    return 1ULL << (63 - square);
}

int to_square(uint64_t bit) {
    return 63 - std::countr_zero(bit);
}

std::vector<int> to_squares(uint64_t bits) {
    std::vector<int> squares;
    for (; bits != 0; bits &= bits - 1) {
        squares.push_back(to_square(bits));
    }
    // countr_zero walks h8 first; return them in reading order
    std::reverse(squares.begin(), squares.end());
    return squares;
}

MinimaxPlayer::Heuristic heuristic_named(const std::string& name) {
    if (name == "mobility") {
        return othello_bot::MinimaxPlayer::mobility;
    }
    if (name == "disc_difference") {
        return othello_bot::MinimaxPlayer::disc_difference;
    }
    throw nb::value_error("heuristic must be 'mobility' or 'disc_difference'");
}

std::optional<othello_bot::MinimaxPlayer::Clock::duration> to_duration(std::optional<double> seconds) {
    if (!seconds) {
        return std::nullopt;
    }
    return std::chrono::duration_cast<othello_bot::MinimaxPlayer::Clock::duration>(
        std::chrono::duration<double>(*seconds)
    );
}

// the bot's move as a square; the GIL is released while it searches
int bot_move(Player& player, const Board& board) {
    if (board.game_over() || board.legal_moves() == 0) {
        throw nb::value_error("no legal move to make");
    }
    uint64_t move = 0;
    {
        nb::gil_scoped_release release;
        move = player.get_move(board);
    }
    return to_square(move);
}

} // namespace


NB_MODULE(_othello_bot, m) {
    m.doc() = "Othello board and bots from othello_bot";

    nb::enum_<Color>(m, "Color")
        .value("BLACK", Color::BLACK)
        .value("WHITE", Color::WHITE);

    m.def("square_name", [](int square) { return othello::util::move_to_string(to_bit(square)); },
        "square"_a, "the square's name, e.g. 19 -> 'd3'");

    nb::class_<Board>(m, "Board")
        .def(nb::init<>(), "the starting position, black to move")
        .def(nb::init<const Board&>(), "a copy of `board`", "board"_a)
        .def("__copy__", [](const Board& b) { return Board(b); })
        .def_prop_ro("turn", &Board::turn, "side to move")
        .def_prop_ro("game_over", &Board::game_over)
        .def_prop_ro("black_count", &Board::black_count)
        .def_prop_ro("white_count", &Board::white_count)
        .def_prop_ro("last_move_caused_pass", &Board::last_move_caused_pass,
            "true if the opponent had to pass after the last move, so the same side moves again")
        .def("at", [](const Board& b, int square) -> std::optional<Color> {
            const uint64_t bit = to_bit(square);
            if (b.black() & bit) return Color::BLACK;
            if (b.white() & bit) return Color::WHITE;
            return std::nullopt;
        }, "square"_a, "the color of the disc on `square`, or None")
        .def("legal_moves", [](const Board& b) { return to_squares(b.legal_moves()); },
            "squares the side to move can play")
        .def("captures", [](const Board& b, int square) { return to_squares(b.captures(to_bit(square))); },
            "square"_a, "squares that playing `square` would flip")
        .def("make_move", [](Board& b, int square) {
            const uint64_t bit = to_bit(square);
            if (b.game_over() || !b.move_valid(bit)) {
                throw nb::value_error("illegal move");
            }
            b.make_move(bit);
        }, "square"_a, "plays `square` for the side to move; passes are applied automatically")
        .def("__str__", &Board::to_string);

    nb::class_<Player>(m, "Player")
        .def("get_move", &bot_move, "board"_a, "the square this player plays on `board`");

    nb::class_<MinimaxPlayer, Player>(m, "MinimaxPlayer")
        .def("__init__", [](MinimaxPlayer* self, int depth, const std::string& heuristic, std::optional<double> time_limit) {
            if (depth < 1) {
                throw nb::value_error("depth must be at least 1");
            }
            new (self) MinimaxPlayer(depth, heuristic_named(heuristic));
            self->set_time_limit(to_duration(time_limit));
        }, "depth"_a = 6, "heuristic"_a = "mobility", "time_limit"_a = nb::none(),
            "alpha-beta minimax; heuristic is 'mobility' or 'disc_difference', time_limit in seconds per move")
        .def_prop_ro("depth", &MinimaxPlayer::depth);

    nb::class_<othello_bot::MCTSPlayer, Player>(m, "MCTSPlayer")
        .def(nb::init<std::size_t, std::optional<uint64_t>>(), "iterations"_a = 10'000, "seed"_a = nb::none(),
            "UCT Monte Carlo tree search with `iterations` random playouts per move")
        .def_prop_ro("iterations", &othello_bot::MCTSPlayer::iterations);

    nb::class_<othello_bot::NetworkPlayer, Player>(m, "NetworkPlayer")
        .def_static("load", [](const std::filesystem::path& path, int depth, std::optional<double> time_limit) {
            if (depth < 1) {
                throw nb::value_error("depth must be at least 1");
            }
            auto player = othello_bot::NetworkPlayer::load(path, depth);
            player->search().set_time_limit(to_duration(time_limit));
            return player;
        }, "path"_a, "depth"_a = 4, "time_limit"_a = nb::none(),
            "a minimax player scored by the network in a .ckpt or .mlfo file from train");
}
