#include <algorithm>
#include <charconv>
#include <cmath>
#include <numbers>
#include <format>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include <othello_bot/train_config.hpp>

namespace othello_bot {

namespace {

std::string_view trim(std::string_view s) {
    const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && space(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && space(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

template <class T>
T parse_number(std::string_view text) {
    text = trim(text);
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        throw std::invalid_argument(std::format("'{}' is not a valid number", text));
    }
    return value;
}

template <class T>
T parse_value(std::string_view text) {
    if constexpr (std::is_same_v<T, std::filesystem::path>) {
        return std::filesystem::path(std::string(trim(text)));
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::string(trim(text));
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
        // comma separated, may be empty
        std::vector<int> values;
        text = trim(text);
        while (!text.empty()) {
            const std::size_t comma = text.find(',');
            values.push_back(parse_number<int>(text.substr(0, comma)));
            if (comma == std::string_view::npos) {
                break;
            }
            text.remove_prefix(comma + 1);
        }
        return values;
    } else {
        return parse_number<T>(text);
    }
}

template <class T>
std::string format_value(const T& value) {
    if constexpr (std::is_same_v<T, std::filesystem::path>) {
        return value.generic_string();
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
        std::string out;
        for (std::size_t i = 0; i < value.size(); ++i) {
            out += (i == 0 ? "" : ",") + std::to_string(value[i]);
        }
        return out;
    } else {
        // shortest representation that reads back to the same value
        return std::format("{}", value);
    }
}

struct Field {
    std::string_view name;
    std::function<void(TrainConfig&, std::string_view)> set;
    std::function<std::string(const TrainConfig&)> get;
};

// access is a generic lambda returning a reference to the field, for both
// const and non-const configs
template <class Access>
Field field(std::string_view name, Access access) {
    using T = std::remove_cvref_t<decltype(access(std::declval<TrainConfig&>()))>;
    return Field{
        name,
        [access](TrainConfig& c, std::string_view v) { access(c) = parse_value<T>(v); },
        [access](const TrainConfig& c) { return format_value<T>(access(c)); },
    };
}

#define OTHELLO_BOT_FIELD(key, member) field(key, [](auto& c) -> auto& { return c.member; })

const std::vector<Field>& fields() {
    static const std::vector<Field> all = {
        OTHELLO_BOT_FIELD("seed", seed),
        OTHELLO_BOT_FIELD("iterations", iterations),
        OTHELLO_BOT_FIELD("games_per_iteration", games_per_iteration),
        OTHELLO_BOT_FIELD("self_play_depth", self_play_depth),
        OTHELLO_BOT_FIELD("random_opening_plies", game.random_opening_plies),
        OTHELLO_BOT_FIELD("epsilon", game.epsilon),
        OTHELLO_BOT_FIELD("outcome_weight", game.outcome_weight),
        OTHELLO_BOT_FIELD("margin_weight", game.margin_weight),
        OTHELLO_BOT_FIELD("threads", threads),
        OTHELLO_BOT_FIELD("buffer_capacity", buffer_capacity),
        OTHELLO_BOT_FIELD("batch_size", batch_size),
        OTHELLO_BOT_FIELD("train_steps_per_iteration", train_steps_per_iteration),
        OTHELLO_BOT_FIELD("learning_rate", learning_rate),
        OTHELLO_BOT_FIELD("lr_schedule", lr_schedule),
        OTHELLO_BOT_FIELD("lr_warmup_iterations", lr_warmup_iterations),
        OTHELLO_BOT_FIELD("lr_decay", lr_decay),
        OTHELLO_BOT_FIELD("lr_decay_every", lr_decay_every),
        OTHELLO_BOT_FIELD("lr_min", lr_min),
        OTHELLO_BOT_FIELD("evaluate_every", evaluate_every),
        OTHELLO_BOT_FIELD("eval_depth", eval_depth),
        OTHELLO_BOT_FIELD("baseline_depths", baseline_depths),
        OTHELLO_BOT_FIELD("eval_openings", eval_openings),
        OTHELLO_BOT_FIELD("eval_random_opening_plies", eval_random_opening_plies),
        OTHELLO_BOT_FIELD("table_bits", table_bits),
        OTHELLO_BOT_FIELD("run_dir", run_dir),
        OTHELLO_BOT_FIELD("checkpoint_every", checkpoint_every),
        OTHELLO_BOT_FIELD("init_model", init_model),
    };
    return all;
}

#undef OTHELLO_BOT_FIELD

void require(bool ok, std::string_view message) {
    if (!ok) {
        throw std::invalid_argument(std::string(message));
    }
}

} // namespace

std::vector<std::string_view> TrainConfig::keys() {
    std::vector<std::string_view> names;
    for (const auto& f : fields()) {
        names.push_back(f.name);
    }
    return names;
}

void TrainConfig::set(std::string_view key, std::string_view value) {
    key = trim(key);
    for (const auto& f : fields()) {
        if (f.name == key) {
            try {
                f.set(*this, value);
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument(std::format("{}: {}", key, e.what()));
            }
            return;
        }
    }
    throw std::invalid_argument(std::format("unknown setting '{}'", key));
}

void TrainConfig::load_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::invalid_argument("could not open config file " + path.string());
    }
    std::string line;
    for (int number = 1; std::getline(file, line); ++number) {
        std::string_view text = line;
        if (const std::size_t hash = text.find('#'); hash != std::string_view::npos) {
            text = text.substr(0, hash);
        }
        text = trim(text);
        if (text.empty()) {
            continue;
        }
        const std::size_t equals = text.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument(
                std::format("{}:{}: expected 'key = value'", path.string(), number)
            );
        }
        try {
            set(text.substr(0, equals), text.substr(equals + 1));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::format("{}:{}: {}", path.string(), number, e.what()));
        }
    }
}

void TrainConfig::parse_args(int argc, const char* const* argv) {
    for (int i = 0; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (!arg.starts_with("--")) {
            throw std::invalid_argument(std::format("unexpected argument '{}'", arg));
        }
        arg.remove_prefix(2);
        std::string_view key = arg;
        std::string_view value;
        if (const std::size_t equals = arg.find('='); equals != std::string_view::npos) {
            key = arg.substr(0, equals);
            value = arg.substr(equals + 1);
        } else {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::format("--{} needs a value", key));
            }
            value = argv[++i];
        }
        if (key == "config") {
            load_file(std::filesystem::path(std::string(value)));
        } else {
            set(key, value);
        }
    }
}

void TrainConfig::validate() const {
    require(iterations > 0, "iterations must be positive");
    require(games_per_iteration > 0, "games_per_iteration must be positive");
    require(self_play_depth >= 1, "self_play_depth must be at least 1");
    require(game.random_opening_plies >= 0, "random_opening_plies must not be negative");
    require(game.epsilon >= 0.0 && game.epsilon <= 1.0, "epsilon must be in [0, 1]");
    require(
        game.outcome_weight >= 0.0f && game.outcome_weight <= 1.0f,
        "outcome_weight must be in [0, 1]"
    );
    require(
        game.margin_weight >= 0.0f && game.margin_weight <= 1.0f,
        "margin_weight must be in [0, 1]"
    );
    require(buffer_capacity > 0, "buffer_capacity must be positive");
    require(batch_size > 0, "batch_size must be positive");
    require(train_steps_per_iteration > 0, "train_steps_per_iteration must be positive");
    require(learning_rate > 0.0f, "learning_rate must be positive");
    require(
        lr_schedule == "constant" || lr_schedule == "step" || lr_schedule == "cosine",
        "lr_schedule must be constant, step or cosine"
    );
    require(lr_decay > 0.0f && lr_decay <= 1.0f, "lr_decay must be in (0, 1]");
    require(lr_decay_every > 0, "lr_decay_every must be positive");
    require(
        lr_min > 0.0f && lr_min <= learning_rate,
        "lr_min must be positive and at most learning_rate"
    );
    require(evaluate_every > 0, "evaluate_every must be positive");
    require(eval_depth >= 1, "eval_depth must be at least 1");
    for (const int depth : baseline_depths) {
        require(depth >= 1, "baseline_depths must all be at least 1");
    }
    require(eval_openings >= 0, "eval_openings must not be negative");
    require(eval_random_opening_plies >= 0, "eval_random_opening_plies must not be negative");
    require(table_bits >= 0 && table_bits <= 30, "table_bits must be in [0, 30]");
    require(checkpoint_every > 0, "checkpoint_every must be positive");
}

float TrainConfig::learning_rate_at(std::size_t iteration) const {
    iteration = std::max<std::size_t>(iteration, 1);
    if (iteration <= lr_warmup_iterations) {
        return learning_rate * static_cast<float>(iteration) /
            static_cast<float>(lr_warmup_iterations);
    }
    // 0-based position within the schedule proper, after the warmup
    const std::size_t k = iteration - lr_warmup_iterations - 1;
    if (lr_schedule == "step") {
        const auto blocks = static_cast<float>(k / lr_decay_every);
        // the decay never goes below lr_min, so the rate stays positive
        return std::max(lr_min, learning_rate * std::pow(lr_decay, blocks));
    }
    if (lr_schedule == "cosine") {
        const std::size_t span = iterations > lr_warmup_iterations + 1
            ? iterations - lr_warmup_iterations - 1
            : 1;
        const double t = std::min(1.0, static_cast<double>(k) / static_cast<double>(span));
        const double cosine = 0.5 * (1.0 + std::cos(std::numbers::pi * t));
        return static_cast<float>(lr_min + (learning_rate - lr_min) * cosine);
    }
    return learning_rate;
}

std::string TrainConfig::to_string() const {
    std::string out;
    for (const auto& f : fields()) {
        out += std::format("{} = {}\n", f.name, f.get(*this));
    }
    return out;
}

} // namespace othello_bot
