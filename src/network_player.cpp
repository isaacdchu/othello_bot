#include <utility>

#include <othello_bot/network_player.hpp>

namespace othello_bot {

NetworkPlayer::NetworkPlayer(ValueNetwork network, int depth, int table_bits) :
network_(std::make_unique<ValueNetwork>(std::move(network))),
search_(depth, network_->heuristic(), table_bits)
{
    // nothing
}

std::unique_ptr<NetworkPlayer> NetworkPlayer::load(
    const std::filesystem::path& path, int depth, int table_bits
) {
    // batch size 1: the network only ever evaluates single positions here
    return std::make_unique<NetworkPlayer>(ValueNetwork::load_any(path, 1), depth, table_bits);
}

uint64_t NetworkPlayer::get_move(const othello::board::Board& board) {
    return search_.get_move(board);
}

} // namespace othello_bot
