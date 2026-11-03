#include <bit>
#include <stdexcept>
#include <string>
#include <string_view>

#include <othello_bot/self_play.hpp>

namespace othello_bot::self_play {

namespace {

uint64_t random_move(uint64_t moves, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> pick(0, std::popcount(moves) - 1);
    for (int skip = pick(rng); skip > 0; --skip) {
        moves &= moves - 1;
    }
    return moves & -moves;
}

// +1 / 0 / -1 for `side`
float outcome(const othello::board::Board& board, othello::Color side) {
    const int margin = board.margin();
    const int black = (margin > 0) - (margin < 0);
    return static_cast<float>(side == othello::Color::BLACK ? black : -black);
}

} // namespace

std::vector<Sample> play_game(
    MinimaxPlayer& player,
    const GameConfig& config,
    std::mt19937_64& rng
) {
    const float lambda = config.outcome_weight;
    if (!(lambda >= 0.0f && lambda <= 1.0f)) {
        throw std::invalid_argument("[self_play::play_game] outcome_weight must be in [0, 1]");
    }
    const bool need_search_value = lambda < 1.0f;

    struct Visited {
        encoding::Planes planes;
        othello::Color side;
        // the search value part of the target, already in [-1, 1]
        float search_value;
    };
    std::vector<Visited> visited;
    std::bernoulli_distribution explore(config.epsilon);

    othello::board::Board board;
    for (int ply = 0; !board.game_over(); ++ply) {
        const uint64_t moves = board.legal_moves();
        if (std::has_single_bit(moves)) {
            // forced: nothing to learn or choose
            board.make_move(moves);
            continue;
        }
        const bool random = ply < config.random_opening_plies || explore(rng);
        uint64_t move = 0;
        float search_value = 0.0f;
        if (!random || need_search_value) {
            move = player.get_move(board);
            if (need_search_value) {
                // the position has several moves, so a search ran and set the value
                search_value = ValueNetwork::target_from_search(*player.last_value());
            }
        }
        if (random) {
            // the search value still labels this position: it describes the
            // position, not the move about to be played
            move = random_move(moves, rng);
        }
        visited.push_back({encoding::planes(board), board.turn(), search_value});
        board.make_move(move);
    }

    std::vector<Sample> samples;
    samples.reserve(visited.size());
    for (const auto& v : visited) {
        const float target = lambda * outcome(board, v.side) + (1.0f - lambda) * v.search_value;
        samples.push_back({v.planes, target});
    }
    return samples;
}

ReplayBuffer::ReplayBuffer(std::size_t capacity) :
capacity_(capacity)
{
    if (capacity_ == 0) {
        throw std::invalid_argument("[ReplayBuffer] capacity must be positive");
    }
}

void ReplayBuffer::add(std::span<const Sample> samples) {
    for (const auto& s : samples) {
        if (samples_.size() == capacity_) {
            samples_.pop_front();
        }
        samples_.push_back(s);
    }
}

std::vector<Sample> ReplayBuffer::sample(std::size_t count, std::mt19937_64& rng) const {
    if (samples_.empty()) {
        throw std::logic_error("[ReplayBuffer::sample] buffer is empty");
    }
    std::uniform_int_distribution<std::size_t> pick(0, samples_.size() - 1);
    std::vector<Sample> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(samples_[pick(rng)]);
    }
    return out;
}

namespace {

constexpr std::string_view BUFFER_HEADER = "replay_buffer";

template <class T>
void write_raw(std::ostream& os, T value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T>
T read_raw(std::istream& is) {
    T value{};
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

} // namespace

void ReplayBuffer::write(std::ostream& os) const {
    os << BUFFER_HEADER << '\n' << samples_.size() << '\n';
    for (const auto& s : samples_) {
        write_raw(os, s.planes.own);
        write_raw(os, s.planes.opp);
        write_raw(os, s.planes.legal);
        write_raw(os, s.target);
    }
    if (!os) {
        throw std::runtime_error("[ReplayBuffer::write] write failed");
    }
}

ReplayBuffer ReplayBuffer::read(std::istream& is, std::size_t capacity) {
    std::string header;
    std::size_t count = 0;
    if (!std::getline(is, header) || header != BUFFER_HEADER || !(is >> count) || is.get() != '\n') {
        throw std::runtime_error("[ReplayBuffer::read] not a replay buffer");
    }
    ReplayBuffer buffer(capacity);
    for (std::size_t i = 0; i < count; ++i) {
        Sample s;
        s.planes.own = read_raw<uint64_t>(is);
        s.planes.opp = read_raw<uint64_t>(is);
        s.planes.legal = read_raw<uint64_t>(is);
        s.target = read_raw<float>(is);
        if (!is) {
            throw std::runtime_error("[ReplayBuffer::read] truncated replay buffer");
        }
        buffer.add(std::span<const Sample>(&s, 1));
    }
    return buffer;
}

MatchResult match(
    othello::player::Player& a,
    othello::player::Player& b,
    int openings,
    int random_opening_plies,
    std::mt19937_64& rng
) {
    MatchResult result;
    for (int o = 0; o < openings; ++o) {
        othello::board::Board opening;
        for (int ply = 0; ply < random_opening_plies && !opening.game_over(); ++ply) {
            opening.make_move(random_move(opening.legal_moves(), rng));
        }
        for (const bool a_black : {true, false}) {
            othello::board::Board board = opening;
            while (!board.game_over()) {
                const bool black_to_move = board.turn() == othello::Color::BLACK;
                auto& player = (black_to_move == a_black) ? a : b;
                board.make_move(player.get_move(board));
            }
            const float r = outcome(board, a_black ? othello::Color::BLACK : othello::Color::WHITE);
            if (r > 0) {
                ++result.wins;
            } else if (r < 0) {
                ++result.losses;
            } else {
                ++result.draws;
            }
        }
    }
    return result;
}

} // namespace othello_bot::self_play
