#include <stdexcept>

#include <othello_bot/encoding.hpp>

namespace othello_bot::encoding {

Planes planes(const othello::board::Board& board) {
    const bool black = board.turn() == othello::Color::BLACK;
    return Planes{
        .own = black ? board.black() : board.white(),
        .opp = black ? board.white() : board.black(),
        .legal = board.legal_moves(),
    };
}

uint64_t transform(uint64_t bitboard, int sym) {
    if (sym < 0 || sym >= SYMMETRIES) {
        throw std::invalid_argument("[encoding::transform] symmetry must be in [0, 8)");
    }
    if (sym == 0) {
        return bitboard;
    }
    // 1..7 are othello's seven non-identity symmetries, in enum order
    return othello::board::Board::transform(
        bitboard, static_cast<othello::board::Board::Symmetry>(sym - 1)
    );
}

Planes transform(const Planes& p, int sym) {
    return Planes{
        .own = transform(p.own, sym),
        .opp = transform(p.opp, sym),
        .legal = transform(p.legal, sym),
    };
}

void encode(const Planes& p, std::span<float> out) {
    if (out.size() != INPUT_SIZE) {
        throw std::invalid_argument("[encoding::encode] output must hold INPUT_SIZE floats");
    }
    const uint64_t channels[CHANNELS] = {p.own, p.opp, p.legal};
    for (std::size_t c = 0; c < CHANNELS; ++c) {
        for (int sq = 0; sq < 64; ++sq) {
            out[sq * CHANNELS + c] = static_cast<float>((channels[c] >> (63 - sq)) & 1);
        }
    }
}

} // namespace othello_bot::encoding
