#ifndef CNN_H
#define CNN_H

#include "../board/board.h"
#include "tensor.h"

#include <string>
#include <iostream>
#include <array>

class CNN {
public:
    CNN();
    CNN(const std::string &model_path);
    void train(const std::string &data_path);
    float predict(const std::string &data_path) const;
    void save_model(const std::string &model_path) const;
private:
    std::pair<Tensor<8, 8, 3>, const int> parse_line(const std::string &line) const;
    void backpropagate(const int &label);
};

#endif // CNN_H