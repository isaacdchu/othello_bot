#ifndef MAIN_H
#define MAIN_H

#include "board/board.h"
#include "player/player.h"
#include "player/human.cpp"
#include "player/randobot.cpp"
#include "utils/utils.h"

#include <iostream>

int main();
void run_game(Player* player_1, Player* player_2, Board& board);

#endif // MAIN_H