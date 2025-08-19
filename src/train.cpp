#include "train.h"

int main() {
    std::cout << "Starting training..." << std::endl;
    const std::string model_path = "model/cnn_model.txt";
    const std::string data_path = "data/sample.txt";
    CNN cnn = CNN(0.001);
    cnn.train(data_path);
    const std::string test_line = "0,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1 4";
    Output output = cnn.predict(CNN::parse_line(test_line).first);
    std::cout << "Policy: " << output.policy.to_string() << std::endl;
    std::cout << "Value: " << output.value << std::endl;
    return 0;
}