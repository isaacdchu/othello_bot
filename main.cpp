// Plays a game of Othello in the terminal.
//
//   ./build/main                      MCTS vs mobility minimax, bot against bot
//   ./build/main MODEL [DEPTH]        you (black) against a trained network
//
// MODEL is any file train writes: a checkpoint (.ckpt, e.g.
// runs/<run>/latest.ckpt or runs/<run>/checkpoints/iter_000100.ckpt) or a
// weights file (.mlfo). DEPTH is how many plies the network searches (default 4).
// Enter moves like d3.

#include <exception>
#include <iostream>
#include <memory>
#include <print>
#include <string>
#include <utility>

#include <othello/othello.hpp>

#include <othello_bot/othello_bot.hpp>

int main(int argc, char** argv) {
    if (argc > 3) {
        std::println(std::cerr, "usage: {} [MODEL [DEPTH]]", argv[0]);
        return 2;
    }

    std::unique_ptr<othello::player::Player> black;
    std::unique_ptr<othello::player::Player> white;
    if (argc == 1) {
        black = std::make_unique<othello_bot::MinimaxPlayer>(11, othello_bot::MinimaxPlayer::mobility);
        white = std::make_unique<othello_bot::MinimaxPlayer>(8, othello_bot::MinimaxPlayer::mobility);
    } else {
        int depth = 4;
        if (argc == 3) {
            try {
                depth = std::stoi(argv[2]);
            } catch (const std::exception&) {
                std::println(std::cerr, "error: DEPTH must be a number, got {}", argv[2]);
                return 2;
            }
        }
        try {
            black = std::make_unique<othello_bot::MinimaxPlayer>(6, othello_bot::MinimaxPlayer::mobility);
            white = othello_bot::NetworkPlayer::load(argv[1], depth);
        } catch (const std::exception& e) {
            std::println(std::cerr, "error: could not load {}: {}", argv[1], e.what());
            return 1;
        }
    }

    othello::game::Game game(std::move(black), std::move(white));
    game.run();
    return 0;
}
