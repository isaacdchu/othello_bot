#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <span>
#include <vector>

#include <othello/othello.hpp>

#include "value_network.hpp"

namespace othello_bot::self_play {

struct GameConfig {
    // the first this many plies are played uniformly at random, so games from
    // a deterministic search still differ from one another
    int random_opening_plies = 8;
    // after the opening, a random legal move is played with this probability
    double epsilon = 0.05;
    // lambda in target = lambda * result + (1 - lambda) * search value, in [0, 1]
    //   1: the game result only (+1 / 0 / -1). Unbiased but noisy, and
    //      exploratory moves late in a game change the label of every
    //      position before them.
    //   0: the player's search value for each position, mapped to [-1, 1] by
    //      ValueNetwork::target_from_search. Low noise, but only as good as
    //      the network doing the search.
    // Below 1, every recorded position is searched, including positions where
    // a random move is then played, so those games take longer to generate.
    float outcome_weight = 1.0f;
    // how the game's result is scored, in [0, 1]:
    //   result = (1 - margin_weight) * win/draw/loss + margin_weight * margin / 64
    // 0 is +1 / 0 / -1 only. The disc margin separates a narrow win from a
    // crushing one, a smoother signal with less noise per sample.
    float margin_weight = 0.0f;
};

// Plays one game with `player` on both sides and returns one sample per
// position where the side to move had a choice, labelled as described by
// config.outcome_weight.
std::vector<Sample> play_game(
    MinimaxPlayer& player,
    const GameConfig& config,
    std::mt19937_64& rng
);

// Fixed-capacity FIFO of samples; the oldest are dropped first.
class ReplayBuffer {
private:
    std::size_t capacity_;
    std::deque<Sample> samples_;

public:
    explicit ReplayBuffer(std::size_t capacity);

    void add(std::span<const Sample> samples);

    // uniformly random samples, with replacement
    std::vector<Sample> sample(std::size_t count, std::mt19937_64& rng) const;

    std::size_t size() const {
        return samples_.size();
    }

    std::size_t capacity() const {
        return capacity_;
    }

    // oldest first
    const std::deque<Sample>& samples() const {
        return samples_;
    }

    // Binary format, for a stream opened in binary mode: a "replay_buffer"
    // line, a line with the sample count, then each sample as own, opp and
    // legal (uint64_t) and target (float) in native byte order. Not portable
    // between machines of different endianness.
    void write(std::ostream& os) const;

    // reads what write() wrote, into a buffer of `capacity`; if the stream
    // holds more samples than that, the oldest are dropped
    static ReplayBuffer read(std::istream& is, std::size_t capacity);
};

struct MatchResult {
    int wins = 0;
    int draws = 0;
    int losses = 0;
    // sum of a's disc margins over every game (positive: a ahead)
    int margin = 0;
    // thinking time and moves played by each side, including forced moves
    double a_seconds = 0.0;
    double b_seconds = 0.0;
    int a_moves = 0;
    int b_moves = 0;

    int games() const {
        return wins + draws + losses;
    }

    // wins count 1, draws 0.5
    double score() const {
        const int n = games();
        return n == 0 ? 0.0 : (wins + 0.5 * draws) / n;
    }

    // adds other's games to this one
    MatchResult& operator+=(const MatchResult& other);
};

// `count` positions, each reached from the start by `random_opening_plies`
// uniformly random moves (fewer if the game ends first). Drawing them from an
// rng seeded the same way gives the same openings that match() plays.
std::vector<othello::board::Board> random_openings(
    int count,
    int random_opening_plies,
    std::mt19937_64& rng
);

// plays `opening` to the end twice, a as black then a as white, and scores
// both games from a's point of view
MatchResult play_pair(
    othello::player::Player& a,
    othello::player::Player& b,
    const othello::board::Board& opening
);

// Makes one player; called once per thread, so every thread has its own.
using PlayerFactory = std::function<std::unique_ptr<othello::player::Player>()>;

// match() over the given openings on up to `threads` threads, each with its
// own pair of players from the factories. Openings are handed out one at a
// time, so with players that keep state between games (e.g. a transposition
// table's move hints) individual games can depend on the thread count; the
// openings and colours played do not.
MatchResult parallel_match(
    const PlayerFactory& make_a,
    const PlayerFactory& make_b,
    std::span<const othello::board::Board> openings,
    int threads
);

// Plays a against b from `openings` random openings of `random_opening_plies`
// plies each. Every opening is played twice with the colours swapped, so a
// lopsided opening counts for both sides equally: 2 * openings games in all,
// scored from a's point of view. Passing an rng seeded the same way each time
// replays the same openings, which keeps results comparable across a run.
MatchResult match(
    othello::player::Player& a,
    othello::player::Player& b,
    int openings,
    int random_opening_plies,
    std::mt19937_64& rng
);

} // namespace othello_bot::self_play

#endif // SELF_PLAY_HPP
