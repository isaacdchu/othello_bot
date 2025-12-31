#include "cnn/tensor.hpp"
#include "cnn/model.hpp"
#include "cnn/cnn.hpp"
#include "cnn/loss.hpp"
#include "cnn/mse_loss.hpp"
#include "cnn/optimizer.hpp"
#include "cnn/sgd.hpp"
#include "dataset/othello_dataset.hpp"
#include "dataset/dataloader.hpp"
#include "utils/utils.hpp"

#include <iostream>
#include <vector>
#include <cstddef>

int main() {
    /*
    // Ideal implementation of training loop
    Model& model = CNN();
    Loss& criterion = MSELoss();
    Optimizer& optimizer = SGD(model.parameters(), 0.001);
    OthelloDataset dataset("data/00.txt", 1000);
    DataLoader data_loader(dataset, 0, 32);

    model.train();
    for (std::size_t i = 0; i < data_loader.size(); i++) {
        auto [inputs, targets] = data_loader.get_batch(i);
        optimizer.zero_grad();
        Tensor outputs = model.forward(inputs);
        Tensor loss = criterion.forward(outputs, targets);
        loss.backward();
        optimizer.step();
        std::cout << "Batch " << i << ", Loss: " << loss.at(0) << std::endl;
    }

    OthelloDataset test_dataset("data/01.txt", 1000);
    DataLoader test_data_loader(test_dataset, 0, 32);
    model.eval();
    for (std::size_t i = 0; i < test_data_loader.size(); i++) {
        auto [inputs, targets] = test_data_loader.get_batch(i);
        Tensor outputs = model.forward(inputs);
        Tensor loss = criterion.forward(outputs, targets);
        std::cout << "Batch " << i << ", Loss: " << loss.at(0) << std::endl;
    }
    */
    return 0;
}