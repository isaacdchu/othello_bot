#include <stdexcept>
#include <utility>

#include <othello_bot/value_network.hpp>

namespace othello_bot {

namespace {

namespace layer = mlfo::layer;
namespace init = mlfo::initializer;

} // namespace

ValueNetwork::ValueNetwork(mlfo::model::Model<float>&& model, float learning_rate) :
model_(std::make_unique<mlfo::model::Model<float>>(std::move(model))),
optimizer_(std::make_unique<mlfo::optimizer::Adam<float>>(*model_, learning_rate)),
learning_rate_(learning_rate),
x_(model_->max_batch_size() * encoding::INPUT_SIZE),
y_(model_->max_batch_size())
{
    // nothing
}

ValueNetwork ValueNetwork::create(uint32_t seed, std::size_t batch_size, float learning_rate) {
    init::Generator generator(seed);
    auto he = std::make_shared<init::He<std::normal_distribution<float>>>(generator);
    auto xavier = std::make_shared<init::Xavier<std::uniform_real_distribution<float>>>(generator);

    std::vector<std::unique_ptr<layer::Layer<float>>> layers;
    // (3, 8, 8) -> (32, 8, 8) x3, 3x3 kernels with padding keep the board size
    for (int i = 0; i < 3; ++i) {
        layers.push_back(std::make_unique<layer::Conv2DLayer<float>>(32, 3, 3, 1, 1, 1, 1, he));
        layers.push_back(std::make_unique<layer::activation::ReLUActivation<float>>());
    }
    // -> (1, 2048) -> 64 -> 1
    layers.push_back(std::make_unique<layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<layer::DenseLayer<float>>(64, he));
    layers.push_back(std::make_unique<layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<layer::DenseLayer<float>>(1, xavier));
    layers.push_back(std::make_unique<layer::activation::TanhActivation<float>>());

    mlfo::model::Model<float> model(
        {encoding::CHANNELS, 8, 8},
        std::move(layers),
        std::make_unique<layer::loss::MSELoss<float>>(),
        batch_size
    );
    model.name("othello_value");
    model.build();
    return ValueNetwork(std::move(model), learning_rate);
}

ValueNetwork ValueNetwork::load(const std::filesystem::path& path, std::size_t batch_size, float learning_rate) {
    return ValueNetwork(mlfo::model::load<float>(path, batch_size), learning_rate);
}

void ValueNetwork::save(const std::filesystem::path& path) const {
    model_->save(path);
}

ValueNetwork ValueNetwork::clone() const {
    return ValueNetwork(model_->clone(), learning_rate_);
}

void ValueNetwork::copy_weights_from(const ValueNetwork& other) {
    model_->copy_parameters_from(*other.model_);
}

float ValueNetwork::evaluate(const othello::board::Board& board) {
    ++evaluations_;
    const std::span<float> x(x_.data(), encoding::INPUT_SIZE);
    encoding::encode(encoding::planes(board), x);
    model_->set_inputs({x});
    // inference only: skips the loss layer, so no labels are needed
    model_->forward(false);
    return model_->outputs().front().get().values()[0];
}

MinimaxPlayer::Heuristic ValueNetwork::heuristic() {
    return [this](const othello::board::Board& board) {
        return HEURISTIC_SCALE * evaluate(board);
    };
}

float ValueNetwork::train_batch_with_symmetries(std::span<const Sample> samples, std::span<const int> symmetries) {
    const std::size_t n = samples.size();
    if (n == 0 || n > batch_size()) {
        throw std::invalid_argument("[ValueNetwork::train_batch] need 1 to batch_size samples");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto planes = encoding::transform(samples[i].planes, symmetries[i]);
        encoding::encode(planes, std::span<float>(x_).subspan(i * encoding::INPUT_SIZE, encoding::INPUT_SIZE));
        y_[i] = samples[i].target;
    }
    model_->set_inputs({std::span<const float>(x_.data(), n * encoding::INPUT_SIZE)});
    model_->set_labels({std::span<const float>(y_.data(), n)});
    model_->forward();

    double loss = 0.0;
    for (const float value : model_->outputs().back().get().values()) {
        loss += value;
    }

    optimizer_->zero_grad();
    model_->backward();
    optimizer_->step();
    return static_cast<float>(loss / n);
}

} // namespace othello_bot
