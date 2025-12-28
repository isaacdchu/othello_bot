#ifndef OTHELLO_DATASET_HPP
#define OTHELLO_DATASET_HPP

#include "dataset.hpp"
#include "../cnn/tensor.hpp"

#include <iostream>
#include <fstream>
#include <string>

class OthelloDataset : public Dataset {
private:
    std::vector<std::pair<Tensor, Tensor>> data_;
    std::pair<Tensor, Tensor> parse_line(const std::string& line) {
        std::vector<float> input_data;
        input_data.reserve(64*3);
        for (std::size_t i = 0; i < 64 * 3; i++) {
            input_data.push_back(line[2 * i] - '0');
        }
        std::vector<float> target_data;
        target_data.reserve(1);
        target_data.push_back(line.back() - '0');
        Tensor input_tensor({3, 8, 8}, input_data);
        Tensor target_tensor({1}, target_data);
        return {input_tensor, target_tensor};
    }
public:
    OthelloDataset(const std::vector<std::pair<Tensor, Tensor>>& data) : data_(data) {}
    OthelloDataset(const std::string& file_path, std::size_t max_samples = 0) {
        std::ifstream data_file(file_path); 
        if (!data_file.is_open()) {
            std::cerr << "Error: Unable to open the file " << file_path << std::endl;
            throw std::runtime_error("File not found");
        }
        std::string line;
        while (std::getline(data_file, line) && (max_samples == 0 || data_.size() < max_samples)) {
            data_.push_back(parse_line(line));
        }
        data_file.close(); 
    }

    std::pair<Tensor, Tensor> get_item(std::size_t index) const override {
        return data_.at(index);
    }

    std::size_t size() const override {
        return data_.size();
    }
};

#endif // OTHELLO_DATASET_HPP