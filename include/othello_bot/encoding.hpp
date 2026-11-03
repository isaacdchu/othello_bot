#ifndef ENCODING_HPP
#define ENCODING_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include <othello/othello.hpp>

namespace othello_bot::encoding {

// A position as the network sees it, always from the point of view of the side
// to move. Kept as bitboards so a replay buffer stays small; it is expanded to
// floats only when a batch is built.
struct Planes {
    uint64_t own = 0;
    uint64_t opp = 0;
    // legal moves of the side to move
    uint64_t legal = 0;
};

// network input shape: (CHANNELS, 8, 8), one channel per Planes field
inline constexpr std::size_t CHANNELS = 3;
inline constexpr std::size_t INPUT_SIZE = CHANNELS * 64;

// the 8 symmetries of the square: 4 rotations, each optionally mirrored
inline constexpr int SYMMETRIES = 8;

Planes planes(const othello::board::Board& board);

// applies symmetry `sym` (0 = identity) to a bitboard / to every plane
uint64_t transform(uint64_t bitboard, int sym);
Planes transform(const Planes& planes, int sym);

// writes INPUT_SIZE floats, channel-major: out[c * 64 + row * 8 + col]
// row 0 / col 0 is a1's corner, matching the board's string layout
void encode(const Planes& planes, std::span<float> out);

} // namespace othello_bot::encoding

#endif // ENCODING_HPP
