#ifndef CNNTRAINER_HPP
#define CNNTRAINER_HPP

#include "cnn.hpp"
#include "adam.hpp"

class CNNTrainer {
public:
    CNNTrainer(CNN& model);
    void train(const std::string& data_path, const size_t epochs);

private:
    CNN& model;
    Adam optimizer;
};

#endif // CNNTRAINER_HPP