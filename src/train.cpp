#include "train.hpp"

int main() {
    std::cout << "Initializing..." << std::endl;
    const std::string model_path = "model/cnn_model.txt";
    const std::string data_path = "data/00.txt";
    CNN cnn = CNN();

    std::cout << "Starting training..." << std::endl;
    cnn.train(data_path, 3000);
    cnn.save_model(model_path);

    std::cout << "Predicting on test line" << std::endl;
    const std::string test_line = "0,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1 4";
    Tensor<8, 8, 3> input_tensor = CNN::parse_line(test_line).first;
    Tensor<8, 8, 1> legal_moves_tensor = Tensor<8, 8, 1>();
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            legal_moves_tensor.set(i, j, 0, input_tensor.at(i, j, 2));
        }
    }
    Output output = cnn.predict(input_tensor);
    std::cout << "Policy: " << output.policy.to_string() << std::endl;
    std::cout << "Value: " << output.value << std::endl;

    Tensor<8, 8, 1> final_policy = output.policy * legal_moves_tensor;
    Board board = Board(input_tensor);
    MCTSNode root = MCTSNode(0, board, nullptr, true); // Black is the root player
    root.initialize_children();
    const unsigned int max_iterations = 10000;
    const unsigned int num_simulations = 10;
    for (unsigned int i = 0; i < max_iterations; i++) {
        auto node = root.select();
        if (node == nullptr) break;
        for (unsigned int j = 0; j < num_simulations; j++) {
            const float result = node->simulate();
            node->backpropagate(result);
        }
    }
    const uint64_t best_move = root.get_best_move();
    std::cout << output.policy.at(__builtin_ctzll(best_move)) << std::endl;
    std::cout << final_policy.to_string() << std::endl;
    return 0;
}