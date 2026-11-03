#ifndef VALUE_NETWORK_HPP
#define VALUE_NETWORK_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include <mlfo/mlfo.hpp>
#include <othello/othello.hpp>

#include "encoding.hpp"
#include "minimax_player.hpp"

namespace othello_bot {

// One training example: a position and the value the network should learn
// for it, from the point of view of the side to move in that position.
struct Sample {
    encoding::Planes planes;
    // in [-1, 1]; a blend of the game result and a search value, see
    // self_play::GameConfig::outcome_weight
    float target = 0.0f;
};

// A CNN that maps a position to a value in [-1, 1] for the side to move,
// wrapped so MinimaxPlayer can use it as a heuristic.
//
// The same mlfo model is used for training (batches of up to batch_size) and
// for evaluating single positions during search; mlfo shrinks a model to the
// number of samples passed in without reallocating.
class ValueNetwork {
private:
    // heap allocated: the optimizer keeps a reference to the model
    std::unique_ptr<mlfo::model::Model<float>> model_;
    std::unique_ptr<mlfo::optimizer::Adam<float>> optimizer_;
    float learning_rate_;
    std::size_t evaluations_ = 0;
    // reused input / label buffers, so evaluating and training don't allocate
    std::vector<float> x_;
    std::vector<float> y_;

    ValueNetwork(
        std::unique_ptr<mlfo::model::Model<float>> model,
        std::unique_ptr<mlfo::optimizer::Adam<float>> optimizer,
        float learning_rate
    );
    ValueNetwork(mlfo::model::Model<float>&& model, float learning_rate);

public:
    // scale applied to the network's [-1, 1] output when used as a heuristic,
    // so it is comparable to the built-in heuristics and far below WIN_SCORE
    static constexpr double HEURISTIC_SCALE = 64.0;

    // maps a MinimaxPlayer value (e.g. last_value()) back to the network's
    // [-1, 1] range: heuristic values are divided by HEURISTIC_SCALE, and
    // finished games (+-WIN_SCORE and beyond) clamp to +-1
    static float target_from_search(double value) {
        return static_cast<float>(std::clamp(value / HEURISTIC_SCALE, -1.0, 1.0));
    }

    // Adam's usual default
    static constexpr float DEFAULT_LEARNING_RATE = 1e-3f;

    // a freshly initialised network
    // TODO: tune the architecture (width, depth, residual blocks once mlfo has an add layer)
    static ValueNetwork create(
        uint32_t seed,
        std::size_t batch_size,
        float learning_rate = DEFAULT_LEARNING_RATE
    );

    // File formats:
    //   weights (.mlfo):     architecture and weights, written by save(). Enough
    //                        to play with, or to start training from scratch.
    //   checkpoint (.ckpt):  also the optimizer's state (Adam's moments and step
    //                        count), written by save_checkpoint(), so training
    //                        continues where it left off.
    // Both are written to a temporary file and renamed into place, so an
    // interrupted save leaves the previous file intact.
    static constexpr std::string_view CHECKPOINT_EXTENSION = ".ckpt";

    // weights only, with a fresh optimizer
    static ValueNetwork load(
        const std::filesystem::path& path,
        std::size_t batch_size,
        float learning_rate = DEFAULT_LEARNING_RATE
    );

    // weights and optimizer state. The learning rate is not taken from the
    // file: it is set to learning_rate, as the training loop owns its schedule.
    static ValueNetwork load_checkpoint(
        const std::filesystem::path& path,
        std::size_t batch_size,
        float learning_rate = DEFAULT_LEARNING_RATE
    );

    // load_checkpoint() for a path ending in CHECKPOINT_EXTENSION, load() otherwise
    static ValueNetwork load_any(
        const std::filesystem::path& path,
        std::size_t batch_size,
        float learning_rate = DEFAULT_LEARNING_RATE
    );

    void save(const std::filesystem::path& path) const;
    void save_checkpoint(const std::filesystem::path& path) const;

    // the checkpoint format on a stream, e.g. to embed in a larger file
    void write_checkpoint(std::ostream& os) const;
    static ValueNetwork read_checkpoint(
        std::istream& is,
        std::size_t batch_size,
        float learning_rate = DEFAULT_LEARNING_RATE
    );

    // an independent copy with the same weights and a fresh optimizer,
    // e.g. to keep as the "best" network
    ValueNetwork clone() const;

    float learning_rate() const {
        return learning_rate_;
    }

    // takes effect from the next train_batch(); Adam's moments are kept
    void set_learning_rate(float learning_rate);

    // overwrites this network's weights with other's (same architecture), in
    // place: no allocation, and heuristics pointing at this network stay valid.
    // The optimizer state is kept.
    void copy_weights_from(const ValueNetwork& other);

    ValueNetwork(ValueNetwork&&) = default;
    ValueNetwork& operator=(ValueNetwork&&) = default;
    ValueNetwork(const ValueNetwork&) = delete;
    ValueNetwork& operator=(const ValueNetwork&) = delete;

    // network output in [-1, 1] for the side to move
    float evaluate(const othello::board::Board& board);

    // a MinimaxPlayer heuristic backed by this network. The network must
    // outlive every player using it, and must not be moved while in use.
    MinimaxPlayer::Heuristic heuristic();

    // one Adam step on samples (at most batch_size of them); each sample gets a
    // random board symmetry drawn from rng. Returns the mean loss before the step.
    template <class Rng>
    float train_batch(std::span<const Sample> samples, Rng& rng);

    std::size_t batch_size() const {
        return model_->max_batch_size();
    }

    // single-position evaluations since construction, for profiling search cost
    std::size_t evaluations() const {
        return evaluations_;
    }

private:
    float train_batch_with_symmetries(std::span<const Sample> samples, std::span<const int> symmetries);
};

template <class Rng>
float ValueNetwork::train_batch(std::span<const Sample> samples, Rng& rng) {
    std::uniform_int_distribution<int> pick(0, encoding::SYMMETRIES - 1);
    std::vector<int> symmetries(samples.size());
    for (int& s : symmetries) {
        s = pick(rng);
    }
    return train_batch_with_symmetries(samples, symmetries);
}

} // namespace othello_bot

#endif // VALUE_NETWORK_HPP
