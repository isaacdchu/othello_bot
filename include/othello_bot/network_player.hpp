#ifndef NETWORK_PLAYER_HPP
#define NETWORK_PLAYER_HPP

#include <filesystem>
#include <memory>

#include <othello/othello.hpp>

#include "minimax_player.hpp"
#include "value_network.hpp"

namespace othello_bot {

// A MinimaxPlayer that scores leaves with a trained value network, and owns
// that network so nothing else has to keep it alive.
//
//   auto bot = othello_bot::NetworkPlayer::load("runs/<run>/latest.ckpt", 4);
//
// Any file train writes works: a checkpoint (.ckpt, e.g. latest.ckpt or
// checkpoints/iter_000100.ckpt) or a weights file (.mlfo). Only the weights
// are used; a checkpoint's optimizer state is loaded and then ignored.
class NetworkPlayer : public othello::player::Player {
private:
    // on the heap so its address, which the heuristic captures, survives moves
    std::unique_ptr<ValueNetwork> network_;
    MinimaxPlayer search_;

public:
    // plays with `network` at `depth` plies (see MinimaxPlayer for table_bits)
    explicit NetworkPlayer(
        ValueNetwork network,
        int depth = 4,
        int table_bits = MinimaxPlayer::DEFAULT_TABLE_BITS
    );

    // the network saved at `path` (.ckpt or .mlfo); throws if it can't be read
    static std::unique_ptr<NetworkPlayer> load(
        const std::filesystem::path& path,
        int depth = 4,
        int table_bits = MinimaxPlayer::DEFAULT_TABLE_BITS
    );

    uint64_t get_move(const othello::board::Board& board) override;

    // the underlying search, e.g. to change its depth or set a time limit
    MinimaxPlayer& search() {
        return search_;
    }

    const MinimaxPlayer& search() const {
        return search_;
    }

    ValueNetwork& network() {
        return *network_;
    }
};

} // namespace othello_bot

#endif // NETWORK_PLAYER_HPP
