#include "train.h"

int main() {
    const std::string model_path = "model/cnn_model.txt";
    for (int i = 0; i <= 9; i++) {
        std::string data_path = "data/000000" + std::to_string(i) + ".txt";
        std::string output_path = "data/0" + std::to_string(i) + ".txt";
        std::cout << "Processing file: " << data_path << std::endl;
        std::string line;
        std::ifstream data(data_path); // Open the file here
        std::ofstream output(output_path); // Open the output file
        if (!data) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        while (std::getline(data, line)) {
            auto [tensor, evaluation] = old_parse_line(line);
            output << tensor.to_string() + " " + std::to_string(evaluation) << std::endl;
        }
    }
    for (int i = 10; i <= 25; i++) {
        std::string data_path = "data/00000" + std::to_string(i) + ".txt";
        std::string output_path = "data/" + std::to_string(i) + ".txt";
        std::cout << "Processing file: " << data_path << std::endl;
        std::string line;
        std::ifstream data(data_path); // Open the file here
        std::ofstream output(output_path); // Open the output file
        if (!data) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        while (std::getline(data, line)) {
            auto [tensor, evaluation] = old_parse_line(line);
            output << tensor.to_string() + " " + std::to_string(evaluation) << std::endl;
        }
    }
    return 0;
}