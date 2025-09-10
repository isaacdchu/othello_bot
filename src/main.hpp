#ifndef MAIN_HPP
#define MAIN_HPP

#include "board/board.hpp"
#include "player/player.hpp"
#include "player/human.cpp"
#include "player/randobot.cpp"
#include "player/otto.cpp"
#include "utils/utils.hpp"

#include <iostream>

int main();
void run_game(Player& player_1, Player& player_2, Board& board);

#endif // MAIN_HPP