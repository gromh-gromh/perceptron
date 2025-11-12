#include "neuron.hpp"
#include <cmath>
#include <iostream>

Neuron::Neuron()
{
    this->value = 0;
    this->is_output = false;
    this->learning_rule = 0;
}

void Neuron::set_value(double value)
{
    if (value <= 1 && value >= 0)
        this->value = value;
    else if (value > 1)
        this->value = 1;
    else
        this->value = 0;
}

void Neuron::set_expected_value(double expected_value)
{
    if (expected_value <= 1 && expected_value >= 0)
        this->expected_value = expected_value;
    else if (expected_value > 1)
        this->expected_value = 1;
    else
        this->expected_value = 0;

    this->is_output = true;
}

double Neuron::get_value()
{
    return this->value;
};

double Neuron::get_expected_value()
{
    return this->expected_value;
}

void Neuron::update_value()
{
    double _value = 0;
    for (size_t input_index = 0; input_index < this->inputs.size(); input_index++)
    {
        Input &input = this->inputs[input_index];
        _value += input.get_value();
    }

    this->value = this->activation(_value);
};

void Neuron::set_input_neurons(std::vector<Neuron> &neurons)
{
    this->inputs.clear();
    this->inputs.reserve(neurons.size());
    
    for(size_t neuron_index = 0; neuron_index < neurons.size(); neuron_index++)
    {
        Input input(neurons[neuron_index]);
        this->inputs.push_back(input);

        // std::cout << this->inputs[neuron_index].get_weight() << std::endl;
    }
};

std::vector<Input> Neuron::get_inputs()
{
    return this->inputs;
}

double Neuron::activation(double value)
{
    return 1 / (1 + exp(-value));
}

void Neuron::update_learning_rule()
{
    if (this->is_output)
        this->learning_rule = this->value * (1 - this->value) * (this->expected_value - this->value);
}

void Neuron::update_learning_rule(double next_learning_rule_sum)
{
    if (!this->is_output)
        this->learning_rule = next_learning_rule_sum * (1 - next_learning_rule_sum);
}

double Neuron::get_learning_rule()
{
    return this->learning_rule;
}

void Neuron::update_weights(double learning_factor)
{
    for (Input &input : this->inputs)
    {
        double new_weight = input.get_weight() + learning_factor * this->learning_rule * input.get_neuron().value;
        input.set_weight(new_weight);
    }
}
