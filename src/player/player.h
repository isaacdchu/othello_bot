#ifndef PLAYER_H
#define PLAYER_H

#include "../board/board.h"
#include "../utils/utils.h"

class Player {
public:
    Player(const std::string& name, bool player_color)
        : name(name), player_color(player_color) {}
    virtual ~Player() = default;
    virtual uint64_t get_move(const Board& board) = 0;
    std::string get_name() const {
        return name;
    }
    bool get_player_color() const {
        return player_color;
    }
private:
    std::string name;
    bool player_color; // true for black, false for white
};

#endif // PLAYER_H