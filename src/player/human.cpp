#ifndef HUMAN_PLAYER_H
#define HUMAN_PLAYER_H

#include "../player/player.h"

class Human : public Player {
public:
    Human(const std::string& name, bool player_color)
        : Player(name, player_color) {}
    virtual ~Human() = default;
    uint64_t get_move(const Board& board) override {
        uint64_t move = 64;
        uint64_t one = 1;
        do {
            std::cout << "Enter your move (0-63): ";
            std::cin >> move;
            if (std::cin.fail() || move < 0 || move >= 64) {
                std::cin.clear(); // Clear the error flag
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                std::cout << "Invalid move. Please enter a number between 0 and 63." << std::endl;
                move = 64; // Reset to an invalid value to continue the loop
                continue;
            }
            if (!(board.get_legal_moves() & (one << move))) {
                std::cout << "Illegal move. Please try again." << std::endl;
                move = 64; // Reset to an invalid value to continue the loop
                continue;
            }
        } while (move >= 64);
        return one << move;
    }
};

#endif // HUMAN_PLAYER_H
