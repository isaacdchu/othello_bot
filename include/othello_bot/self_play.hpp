#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
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
};

// Plays one game with `player` on both sides and returns one sample per
// position where the side to move had a choice, labelled as described by
// config.outcome_weight.
//
// TODO: disc margin / 64 instead of win/draw/loss as the result part (smoother signal)
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

    // TODO: save / load to disk so a run can be resumed
};

struct MatchResult {
    int wins = 0;
    int draws = 0;
    int losses = 0;

    // wins count 1, draws 0.5
    double score() const {
        const int games = wins + draws + losses;
        return games == 0 ? 0.0 : (wins + 0.5 * draws) / games;
    }
};

// `games` games of a against b, alternating colours, from a's point of view.
// `random_opening_plies` random moves start each game so deterministic players
// don't replay the same game.
MatchResult match(
    othello::player::Player& a,
    othello::player::Player& b,
    int games,
    int random_opening_plies,
    std::mt19937_64& rng
);

} // namespace othello_bot::self_play

#endif // SELF_PLAY_HPP
