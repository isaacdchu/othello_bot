#include "cnn/cnn.h"

int main() {
    const std::string model_path = "model/cnn_model.txt";
    const std::string data_path = "data/01.txt";
    CNN cnn = CNN(model_path);
    Metrics metrics = cnn.evaluate(data_path, 100);
    std::cout << "Policy Loss: " << metrics.policy_loss << std::endl;
    std::cout << "Value Loss: " << metrics.value_loss << std::endl;
    return 0;
}
