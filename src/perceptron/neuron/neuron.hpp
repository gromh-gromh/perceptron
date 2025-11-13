#pragma once
#include <stddef.h>
#include <vector>
#include <cstdint>
#include <functional>
#include "../input/input.hpp"

class Input;

class Neuron{
    public:
        Neuron();
        void set_value(double value);
        void set_expected_value(double expected_value);
        double get_value();
        double get_expected_value();
        void update_value();
        void set_input_neurons(std::vector<Neuron> &neurons);
        std::vector<Input> &get_inputs();
        static double activation(double value);
        void update_learning_rule();
        void update_learning_rule(double next_learning_rule_sum);
        double get_learning_rule();
        void update_weights(double learning_factor);

    private:
        double value;
        bool is_output;
        double expected_value;
        double learning_rule;
        std::vector<Input> inputs;
};