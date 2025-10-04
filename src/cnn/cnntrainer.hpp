#ifndef CNNTRAINER_HPP
#define CNNTRAINER_HPP

#include "cnn.hpp"
#include "adam.hpp"

template<size_t num_filters>
class CNNTrainer {
private:
    CNN<num_filters> &model;
    Adam optimizer;
public:
    CNNTrainer(CNN<num_filters>& model, Adam optimizer = Adam(0.001f, 0.9f, 0.999f)) : model(model), optimizer(optimizer) {};
    void train(const std::string& data_path, const size_t epochs);
};

#endif // CNNTRAINER_HPP