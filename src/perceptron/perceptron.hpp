#pragma once
#include <stddef.h>
#include <vector>
#include <cstdint>
#include <functional>
#include "neuron/neuron.hpp"
#include "input/input.hpp"

class Perceptron {
    public:
        Perceptron
        (
            size_t input_size,
            size_t output_size,
            size_t layer_count,
            size_t hidden_layer_size,
            double max_error,
            double min_learning_factor,
            double max_learning_factor
        );
        void set_input(std::vector<double> input);
        void set_expected_output(std::vector<double> expected_output);
        void set_weights(std::vector<std::vector<double>> &weights);
        std::vector<std::vector<double>> get_weights();
        double get_error();
        std::vector<double> get_output();
        void train();
        void run();
        void debug_print_neuron_values();


    private:
        void initialize_neurons();
        void reset_neurons();
        void calculate_neurons();
        void update_error();
        void update_learning_factor();
        void update_learning_rules();
        void update_weights();

        size_t input_size;
        size_t output_size;
        size_t layer_count;
        size_t hidden_layer_size;

        std::vector<double> input;
        std::vector<double> expected_output;

        double error;
        double max_error;
        double min_learning_factor;
        double max_learning_factor;
        double learning_factor;
        std::vector<std::vector<Neuron>> neuron_layers;
};