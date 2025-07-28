#ifndef HUMAN_PLAYER_H
#define HUMAN_PLAYER_H

#include "../player/player.h"

class Human : public Player {
public:
    Human(const std::string& name, bool player_color)
        : Player(name, player_color) {}
    virtual ~Human() = default;
    uint64_t get_move(const Board& board) override {
        std::string input;
        std::cout << get_name() << ", enter your move (a1 - h8): ";
        std::cin >> input;

        // Convert input square to move
        uint64_t move = square_to_move(input);
        if (move == 0) {
            std::cerr << "Invalid move. Please try again." << std::endl;
            return get_move(board); // Retry if the move is invalid
        }

        // Check if the move is legal
        if (!(board.get_legal_moves() & move)) {
            std::cerr << "Illegal move. Please try again." << std::endl;
            return get_move(board); // Retry if the move is illegal
        }

        return move; // Return the valid move
    }
};

#endif // HUMAN_PLAYER_H
