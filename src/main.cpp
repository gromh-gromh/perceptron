#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "perceptron/perceptron.hpp"

constexpr size_t input_size = 49;
constexpr size_t output_size = 3;
constexpr size_t layer_count = 3;
constexpr size_t hidden_layer_size = 21;
constexpr double max_error = 0.1;
constexpr double min_learning_factor = 0.1;
constexpr double max_learning_factor = 0.3;

constexpr double learning_barrier = 0.001;
constexpr double max_barrier = 3;

constexpr size_t learning_epoch_amount = 130;

constexpr std::string_view training_directory = "src/data/train";
constexpr std::string_view validation_directory = "src/data/validate";
std::string model_file = "src/data/model/model.txt";

struct bitImage
{
    std::vector<double> type;
    std::vector<double> data;
};

struct model
{
    size_t input_size;
    size_t output_size;
    size_t layer_count;
    size_t hidden_layer_size;
    std::vector<std::vector<double>> weights;
};

bitImage readImage(auto &path)
{
    bitImage image;

    image.type.resize(output_size);
    image.data.resize(input_size);

    std::ifstream file(path);
    file >> image.type[0] >> image.type[1] >> image.type[2];
    for (size_t i = 0; i < input_size; i++)
        file >> image.data[i];
    file.close();

    return image;
}

void save_model(std::vector<std::vector<double>> &weights, std::string file_path)
{
    std::ofstream file;
    file.open(file_path);

    file << input_size << " ";
    file << output_size << " ";
    file << layer_count << " ";
    file << hidden_layer_size << std::endl;

    for(std::vector<double> &neuron_weights : weights)
    {
        for(double weight : neuron_weights)
        {
            file << weight << " ";
        }
        file << std::endl;
    }
    file.close();
}

model load_model(std::string file_path)
{
    model perceptron_model;
    std::ifstream file(file_path);

    file >> perceptron_model.input_size;
    file >> perceptron_model.output_size;
    file >> perceptron_model.layer_count;
    file >> perceptron_model.hidden_layer_size;

    size_t hidden_neurons_count = (perceptron_model.layer_count - 2) * perceptron_model.hidden_layer_size;
    size_t neuron_with_inputs_count = perceptron_model.output_size + hidden_neurons_count;

    perceptron_model.weights.reserve(neuron_with_inputs_count);

    for (size_t neuron_index = 0; neuron_index < neuron_with_inputs_count; neuron_index++)
    {
        std::vector<double> neuron_weights;
        double weight;
        size_t weight_count;

        if(neuron_index < hidden_layer_size)
            weight_count = perceptron_model.input_size;
        else
            weight_count = perceptron_model.hidden_layer_size;

        for(size_t weight_index = 0; weight_index < weight_count; weight_index++)
        {
            file >> weight;
            neuron_weights.push_back(weight);
        }

        perceptron_model.weights.push_back(neuron_weights);
    }

    return perceptron_model;
}

void printSeparator()
{
    std::cout << std::endl << "===========================================================" << std::endl << std::endl;
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

void validate(Perceptron &perceptron, model &perceptron_model, bool verbose = false)
{
    double mean_error;
    int file_count = 0;
    int success_count = 0;

    for (auto &validation_path : std::filesystem::directory_iterator(validation_directory))
    {
        bitImage image = readImage(validation_path.path());
        perceptron.set_input(image.data);
        perceptron.set_expected_output(image.type);
        perceptron.set_weights(perceptron_model.weights);

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

    std::cout << "Validation ended with " << (double)(success_count) / (double)(file_count) * 100 << "% of correct results" << std::endl;
    std::cout << "Mean error on validation is " << mean_error << std::endl;
}

int main(void){
    Perceptron training_perceptron(
        input_size, 
        output_size, 
        layer_count, 
        hidden_layer_size, 
        max_error,
        min_learning_factor,
        max_learning_factor
    );

    train(training_perceptron);

    std::vector<std::vector<double>> weights = training_perceptron.get_weights();
    save_model(weights, model_file);

    model perceptron_model = load_model(model_file);

    Perceptron validation_perceptron(
        perceptron_model.input_size, 
        perceptron_model.output_size, 
        perceptron_model.layer_count, 
        perceptron_model.hidden_layer_size, 
        max_error,
        min_learning_factor,
        max_learning_factor
    );

    validate(validation_perceptron, perceptron_model);

    return 0;
}