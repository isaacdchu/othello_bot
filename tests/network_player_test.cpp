#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include <gtest/gtest.h>

#include <othello/othello.hpp>
#include <othello_bot/othello_bot.hpp>

using othello_bot::MinimaxPlayer;
using othello_bot::NetworkPlayer;
using othello_bot::ValueNetwork;

namespace {

// a fresh, empty directory under the system temp directory, removed on destruction
class TempDir {
private:
    std::filesystem::path path_;

public:
    explicit TempDir(const std::string& name) :
    path_(std::filesystem::temp_directory_path() / name)
    {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const {
        return path_;
    }
};

} // namespace

TEST(NetworkPlayer, LoadedPlayerMatchesMinimaxWithSameNetwork) {
    TempDir dir("othello_bot_network_player_test");
    auto net = ValueNetwork::create(7, 4);
    net.save(dir.path() / "net.mlfo");
    net.save_checkpoint(dir.path() / "net.ckpt");

    MinimaxPlayer reference(3, net.heuristic());
    auto from_weights = NetworkPlayer::load(dir.path() / "net.mlfo", 3);
    auto from_checkpoint = NetworkPlayer::load(dir.path() / "net.ckpt", 3);

    // play a game with the reference, checking both loaded players agree on every move
    othello::board::Board board;
    while (!board.game_over()) {
        const uint64_t move = reference.get_move(board);
        ASSERT_TRUE(board.move_valid(move));
        EXPECT_EQ(from_weights->get_move(board), move);
        EXPECT_EQ(from_checkpoint->get_move(board), move);
        board.make_move(move);
    }
}

TEST(NetworkPlayer, SurvivesBeingMoved) {
    NetworkPlayer original(ValueNetwork::create(3, 1), 2);
    NetworkPlayer moved = std::move(original);
    othello::board::Board board;
    EXPECT_TRUE(board.move_valid(moved.get_move(board)));
    EXPECT_EQ(moved.search().depth(), 2);
}

TEST(NetworkPlayer, MissingFileThrows) {
    EXPECT_ANY_THROW(NetworkPlayer::load("does/not/exist.ckpt"));
}
