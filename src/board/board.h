#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
#include <string>

class Board {
public:
    Board();
    const std::string pretty_print() const;
    Board deep_copy() const;

private:
    struct State {
        uint64_t black;
        uint64_t white;
    };
    State state;
    bool current_player; // true for black, false for white
};
#endif // BOARD_H
