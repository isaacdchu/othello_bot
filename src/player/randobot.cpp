#ifndef RANDOBOT_H
#define RANDOBOT_H

#include "../player/player.h"
#include <vector>
#include <cstdlib> // for rand()

class RandoBot : public Player
{
public:
    RandoBot(const std::string &name, bool player_color)
        : Player(name, player_color) {}
    virtual ~RandoBot() = default;
    uint64_t get_move(const Board &board) override {
        // Get all legal moves for the current player
        uint64_t legal_moves = board.get_legal_moves();
        // If there are no legal moves, return 0 (indicating no move)
        if (legal_moves == 0) {
            return 0;
        }
        // Randomly select one of the legal moves
        std::vector<uint64_t> moves;
        for (uint64_t move = 1; move <= 64; ++move) {
            if (legal_moves & (1ULL << (move - 1))) {
                moves.push_back(1ULL << (move - 1));
            }
        }
        // If there are no valid moves, return 0
        if (moves.empty()) {
            return 0;
        }
        // Select a random move from the list of legal moves
        size_t random_index = rand() % moves.size();
        return moves[random_index];
    }
};

#endif // HUMAN_PLAYER_H
