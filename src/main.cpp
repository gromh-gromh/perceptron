#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "perceptron/perceptron.hpp"

constexpr size_t input_size = 49;
constexpr size_t output_size = 3;
constexpr size_t layer_count = 3;
constexpr size_t hidden_layer_size = 20;
constexpr double max_error = 0.1;
constexpr double min_learning_factor = 0.1;
constexpr double max_learning_factor = 0.3;

constexpr double learning_barrier = 0.001;
constexpr double max_barrier = 3;

constexpr size_t learning_epoch_amount = 200;

constexpr std::string_view training_directory = "src/data/train";
constexpr std::string_view validation_directory = "src/data/validate";

struct bitImage
{
    std::vector<double> type;
    std::vector<double> data;
};

void printSeparator()
{
    std::cout << std::endl << "===========================================================" << std::endl << std::endl;
}

bitImage readImage(auto &path)
{
    bitImage image;

    image.type.resize(output_size);
    image.data.resize(input_size);

    std::ifstream file(path);
    file >> image.type[0] >> image.type[1] >> image.type[2];
    for (size_t i = 0; i < input_size; i++)
        file >> image.data[i];

    return image;
}

void printVector(std::vector<double> vector)
{
    for (double value : vector)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl;
}

void train(Perceptron &perceptron, bool verbose = false)
{
    double old_mean_error = 10;
    double count_barrier = 0;
    double mean_error;

    size_t epoch;
    clock_t start = clock();

    for (epoch = 1; epoch <= learning_epoch_amount; epoch++)
    {
        mean_error = 0;
        int file_count = 0;

        for (auto &train_path : std::filesystem::directory_iterator(training_directory))
        {
            bitImage image = readImage(train_path.path());

            perceptron.set_input(image.data);
            perceptron.set_expected_output(image.type);

            perceptron.train();

            mean_error += perceptron.get_error();
            file_count++;

            if (verbose)
            {
                printSeparator();
                std::cout << "Epoch: " << epoch << " | Training on: " << train_path.path().filename().generic_string() << std::endl << std::endl;
                std::cout << "Expected output: ";
                printVector(image.type);
                std::cout << std::endl;
                std::cout << "Actual output: ";
                printVector(perceptron.get_output());
                std::cout << std::endl;
                std::cout << "Error: " << perceptron.get_error() << std::endl;
                printSeparator();
            }
        }
        
        mean_error /= file_count;

        if (abs(old_mean_error - mean_error) < learning_barrier || mean_error > old_mean_error)
            count_barrier++;
        else
            count_barrier = 0;

        old_mean_error = mean_error;

        if (count_barrier == max_barrier && mean_error < max_error) {
            break;
        }
    }
    std::cout << "Train ended on epoch " << epoch - 1 << ", time: " << (double)(clock() - start)/CLOCKS_PER_SEC << std::endl;
    std::cout << "Mean error on train is " << mean_error << std::endl;
}

void validate(Perceptron &perceptron, bool verbose = false)
{
    double mean_error;
    int file_count = 0;
    int success_count = 0;

    size_t epoch;

    for (auto &validation_path : std::filesystem::directory_iterator(validation_directory))
    {
        bitImage image = readImage(validation_path.path());

        perceptron.set_input(image.data);
        perceptron.set_expected_output(image.type);

        perceptron.run();

        if (perceptron.get_error() <= max_error)
            success_count++;

        mean_error += perceptron.get_error();
        file_count++;

        if (verbose) 
        {
            printSeparator();
            std::cout << "Validating on: " << validation_path.path().filename().generic_string() << std::endl << std::endl;
            std::cout << "Expected output: ";
            printVector(image.type);
            std::cout << std::endl;
            std::cout << "Actual output: ";
            printVector(perceptron.get_output());
            std::cout << std::endl;
            std::cout << "Error: " << perceptron.get_error() << std::endl;
            printSeparator();
        }
    }

    mean_error /= file_count;

    std::cout << "Validation ended with " << (double)(success_count / file_count) * 100 << "% of correct results" << std::endl;
    std::cout << "Mean error on validation is " << mean_error << std::endl;
}

int main(void){
    Perceptron perceptron(
        input_size, 
        output_size, 
        layer_count, 
        hidden_layer_size, 
        max_error,
        min_learning_factor,
        max_learning_factor
    );

    train(perceptron);
    validate(perceptron, true);

    return 0;
}