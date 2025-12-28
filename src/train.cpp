#include "cnn/cnn.hpp"
#include "dataset/othello_dataset.hpp"
#include "dataset/dataloader.hpp"
#include "utils/utils.hpp"

#include <iostream>
#include <vector>
#include <cstddef>

int main() {
    OthelloDataset dataset("data/00.txt", 1000);
    DataLoader data_loader(dataset, 0, 32);
    for (std::size_t i = 0; i < data_loader.size(); i++) {
        auto [inputs, targets] = data_loader.get_batch(i);
        
    }
    return 0;
}