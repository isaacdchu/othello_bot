#include <stdexcept>

#include <othello_bot/encoding.hpp>

namespace othello_bot::encoding {

namespace {

// the board stores square (row, col) at bit 63 - (row * 8 + col)
constexpr uint64_t bit(int row, int col) {
    return 1ULL << (63 - (row * 8 + col));
}

} // namespace

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
    // TODO: replace with flip / rotate bit tricks if this shows up in profiles
    uint64_t out = 0;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            if ((bitboard & bit(row, col)) == 0) {
                continue;
            }
            int r = row;
            int c = col;
            if (sym & 4) {
                c = 7 - c;  // mirror left-right
            }
            for (int i = 0; i < (sym & 3); ++i) {
                const int t = r;  // rotate 90 degrees clockwise
                r = c;
                c = 7 - t;
            }
            out |= bit(r, c);
        }
    }
    return out;
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
            out[c * 64 + sq] = static_cast<float>((channels[c] >> (63 - sq)) & 1);
        }
    }
}

} // namespace othello_bot::encoding
