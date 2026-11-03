#ifndef RUN_STATE_HPP
#define RUN_STATE_HPP

#include <cstddef>
#include <filesystem>
#include <random>

#include "self_play.hpp"
#include "value_network.hpp"

namespace othello_bot {

// Everything the training loop needs to continue a run after its last
// completed iteration, so that a resumed run plays the same games and takes
// the same training steps as one that was never interrupted.
//
// The config is not part of it: train keeps it in the run's config.txt.
struct RunState {
    // the last completed iteration (1-based)
    std::size_t iteration;
    // the generator used for self-play and for sampling training batches
    std::mt19937_64 rng;
    self_play::ReplayBuffer buffer;
    // weights and optimizer state
    ValueNetwork network;
};

// Writes one file, atomically (see write_atomically), holding a header line,
// the iteration, the rng state, the replay buffer (ReplayBuffer::write) and the
// network's checkpoint (ValueNetwork::write_checkpoint), in that order.
void save_run_state(
    const std::filesystem::path& path,
    std::size_t iteration,
    const std::mt19937_64& rng,
    const self_play::ReplayBuffer& buffer,
    const ValueNetwork& network
);

// Reads a file written by save_run_state. The replay buffer gets
// buffer_capacity (keeping the newest samples if it shrank) and the network
// batch_size and learning_rate, as these come from the config, which may
// differ from the one the state was saved under.
RunState load_run_state(
    const std::filesystem::path& path,
    std::size_t buffer_capacity,
    std::size_t batch_size,
    float learning_rate
);

} // namespace othello_bot

#endif // RUN_STATE_HPP
