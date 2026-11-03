#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <othello_bot/files.hpp>
#include <othello_bot/run_state.hpp>

namespace othello_bot {

namespace {

constexpr std::string_view HEADER = "othello_bot run state 1";

// reads a "<key> <rest>" line and returns <rest>
std::string read_field(std::istream& is, std::string_view key) {
    std::string line;
    if (!std::getline(is, line) || !line.starts_with(key) || line.size() <= key.size() ||
        line[key.size()] != ' ') {
        throw std::runtime_error("[load_run_state] expected '" + std::string(key) + "'");
    }
    return line.substr(key.size() + 1);
}

} // namespace

void save_run_state(
    const std::filesystem::path& path,
    std::size_t iteration,
    const std::mt19937_64& rng,
    const self_play::ReplayBuffer& buffer,
    const ValueNetwork& network
) {
    write_atomically(path, [&](const std::filesystem::path& tmp) {
        // binary: the replay buffer is raw bytes, and must not be newline-translated
        std::ofstream file(tmp, std::ios::binary);
        if (!file) {
            throw std::runtime_error("[save_run_state] could not create " + tmp.string());
        }
        file << HEADER << '\n';
        file << "iteration " << iteration << '\n';
        file << "rng " << rng << '\n';
        buffer.write(file);
        network.write_checkpoint(file);
        file.close();
        if (!file) {
            throw std::runtime_error("[save_run_state] failed writing " + tmp.string());
        }
    });
}

RunState load_run_state(
    const std::filesystem::path& path,
    std::size_t buffer_capacity,
    std::size_t batch_size,
    float learning_rate
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("[load_run_state] could not open " + path.string());
    }
    std::string header;
    if (!std::getline(file, header) || header != HEADER) {
        throw std::runtime_error("[load_run_state] " + path.string() + " is not a run state file");
    }

    std::size_t iteration = 0;
    {
        std::istringstream in(read_field(file, "iteration"));
        if (!(in >> iteration) || iteration == 0) {
            throw std::runtime_error("[load_run_state] bad iteration");
        }
    }
    std::mt19937_64 rng;
    {
        std::istringstream in(read_field(file, "rng"));
        if (!(in >> rng)) {
            throw std::runtime_error("[load_run_state] bad rng state");
        }
    }
    auto buffer = self_play::ReplayBuffer::read(file, buffer_capacity);
    auto network = ValueNetwork::read_checkpoint(file, batch_size, learning_rate);
    return RunState{
        .iteration = iteration,
        .rng = rng,
        .buffer = std::move(buffer),
        .network = std::move(network),
    };
}

} // namespace othello_bot
