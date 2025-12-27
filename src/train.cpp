#include "board/board.hpp"
#include "cnn/cnn.hpp"
#include "dataset/othello_dataset.hpp"
#include "utils/utils.hpp"

#include <iostream>
#include <vector>
#include <cstddef>

int main() {
    OthelloDataset dataset("data/00.txt", 1000);
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    for (std::size_t i = 0; i < dataset.size(); i++) {
        auto [input, target] = dataset.get_item(i);
        
    }
    return 0;
}