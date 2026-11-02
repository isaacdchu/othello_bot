#include <memory>
#include <iostream>
#include <print>

#include <mlfo/mlfo.hpp>
#include <othello/othello.hpp>

#include <othello_bot/othello_bot.hpp>

int main() {
    std::println("size of board class: {}", sizeof(othello::board::Board));
    othello::game::Game game(
        std::make_unique<othello_bot::MCTSPlayer>(20000),
        std::make_unique<othello_bot::MinimaxPlayer>(8, othello_bot::MinimaxPlayer::mobility)
    );
    game.run();
    return 0;
}
