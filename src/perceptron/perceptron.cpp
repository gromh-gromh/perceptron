#include <cmath>
#include "perceptron.hpp"
#include <algorithm>
#include <vector>
#include <iostream>

Perceptron::Perceptron
(
    size_t input_size,
    size_t output_size,
    size_t layer_count,
    size_t intermediate_layer_size,
    double max_error,
    double min_learning_factor,
    double max_learning_factor
) : 
    input_size(input_size), 
    output_size(output_size), 
    layer_count(layer_count), 
    hidden_layer_size(intermediate_layer_size),
    max_error(max_error),
    min_learning_factor(min_learning_factor),
    max_learning_factor(max_learning_factor) {};

void Perceptron::set_input(std::vector<double> input)
{
    this->input = input;
}

void Perceptron::set_expected_output(std::vector<double> expected_output)
{
    this->expected_output = expected_output;
}

void Perceptron::set_weights(std::vector<std::vector<double>> &weights)
{ 
    this->initialize_neurons();

    size_t neuron_weight_index = 0;

    for(size_t layer_index = 1; layer_index < this->layer_count; layer_index++)
    {
        std::vector<Neuron> &layer = this->neuron_layers[layer_index];

        for(size_t neuron_index = 0; neuron_index < layer.size(); neuron_index++)
        {
            Neuron &neuron = layer[neuron_index];
            std::vector<Input> &inputs = neuron.get_inputs();

            for(size_t input_index = 0; input_index < neuron.get_inputs().size(); input_index++)
            {
                Input &input = neuron.get_inputs()[input_index];
                input.set_weight(weights[neuron_weight_index][input_index]);
            }
            neuron_weight_index++;
        }
    }
}

std::vector<std::vector<double>> Perceptron::get_weights()
{
    std::vector<std::vector<double>> weights;

    for(size_t layer_index = 1; layer_index < this->layer_count; layer_index++)
    {
        std::vector<Neuron> &layer = this->neuron_layers[layer_index];

        for(Neuron &neuron : layer)
        {
            std::vector<double> neuron_weights;

            for(Input &input : neuron.get_inputs())
            {
                neuron_weights.push_back(input.get_weight());
            }

            weights.push_back(neuron_weights);
        }
    }

    return weights;
}

double Perceptron::get_error()
{
    return this->error;
}

std::vector<double> Perceptron::get_output()
{
    std::vector<double> output(this->output_size);
    for (size_t neuron_index = 0; neuron_index < this->output_size; neuron_index++)
    {
        output[neuron_index] =  this->neuron_layers[this->layer_count - 1][neuron_index].get_value();
    }
    return output;
}

void Perceptron::train()
{
    if (this->neuron_layers.size() == this->layer_count)
        this->reset_neurons();
    else 
        this->initialize_neurons();

    this->calculate_neurons();

    this->update_learning_factor();
    this->update_error();

    if (this->error > max_error)
    {
        this->update_learning_rules();
        this->update_weights();
    }
}

void Perceptron::run()
{
    if (this->neuron_layers.size() == this->layer_count)
        this->reset_neurons();
    else 
        this->initialize_neurons();

    this->calculate_neurons();
    this->update_error();
}

void Perceptron::debug_print_neuron_values()
{   
    std::cout << "Neuron values" << std::endl;
    for (size_t layer_index = 0; layer_index < this->layer_count; layer_index++)
    {
        for (Neuron &neuron : this->neuron_layers[layer_index])
        {
            std::cout << neuron.get_value() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "(";
    for (size_t neuron_index = 0; neuron_index < this->output_size; neuron_index++)
    {
        std::cout << this->neuron_layers[layer_count - 1][neuron_index].get_expected_value();
        if (neuron_index < this->output_size - 1)
            std::cout << " ";
    }
    std::cout << ")" << std::endl;
}

void Perceptron::initialize_neurons()
{
    this->neuron_layers.resize(this->layer_count);

    for (size_t layer_index = 0; layer_index < this->layer_count; layer_index++)
    {
        if (layer_index == 0) 
        {
            this->neuron_layers[layer_index].resize(this->input_size);

            for(size_t neuron_index = 0; neuron_index < this->input_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_value(this->input[neuron_index]);
            }
        }
        else if (layer_index == this->layer_count - 1)
        {
            this->neuron_layers[layer_index].resize(this->output_size);

            for(size_t neuron_index = 0; neuron_index < this->output_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_expected_value(this->expected_output[neuron_index]);
                this->neuron_layers[layer_index][neuron_index].set_input_neurons(this->neuron_layers[layer_index - 1]);
            }
        }
        else 
        {
            this->neuron_layers[layer_index].resize(this->hidden_layer_size);

            for(size_t neuron_index = 0; neuron_index < this->hidden_layer_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_input_neurons(this->neuron_layers[layer_index - 1]);
            }
        }
    }
}

void Perceptron::reset_neurons()
{
    for (size_t layer_index = 0; layer_index < this->layer_count; layer_index++)
    {
        if (layer_index == 0)
        {
            for(size_t neuron_index = 0; neuron_index < this->input_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_value(this->input[neuron_index]);
            }
        }
        else if (layer_index == this->layer_count - 1)
        {
            for(size_t neuron_index = 0; neuron_index < this->output_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_expected_value(this->expected_output[neuron_index]);
            }
        }
        else 
        {
            for(size_t neuron_index = 0; neuron_index < this->hidden_layer_size; neuron_index++)
            {
                this->neuron_layers[layer_index][neuron_index].set_value(0);
            }
        }
    }
}

void Perceptron::calculate_neurons()
{
    for (size_t layer_index = 1; layer_index < this->layer_count; layer_index++)
    {
        for (Neuron &neuron : this->neuron_layers[layer_index])
        {
            neuron.update_value();
        }
    }
}

void Perceptron::update_error()
{
    this->error = 0;
    for (size_t output_index = 0; output_index < this->output_size; output_index++)
    {
        this->error += fabs(this->expected_output[output_index] - this->neuron_layers[this->layer_count - 1][output_index].get_value());
    }

    this->error/=2;
}

void Perceptron::update_learning_factor()
{
    double mean_error = 2 * this->error / this->output_size;

    this->learning_factor = mean_error * (this->max_learning_factor - this->min_learning_factor) + this->min_learning_factor;
}

void Perceptron::update_learning_rules()
{
    for (size_t layer_index = this->layer_count; layer_index-- > 0;)
    {
        if (layer_index == this->layer_count - 1)
        {
            for (Neuron &neuron : this->neuron_layers[layer_index])
            {
                neuron.update_learning_rule();
            }
        }
        else {
            double next_learning_rule_sum = 0;

            for (size_t neuron_index = 0; neuron_index < this->neuron_layers[layer_index].size(); neuron_index++)
            {
                Neuron &neuron = this->neuron_layers[layer_index][neuron_index];
                for (Neuron &next_layer_neuron : this->neuron_layers[layer_index + 1])
                {
                    next_learning_rule_sum += next_layer_neuron.get_learning_rule() * next_layer_neuron.get_inputs()[neuron_index].get_weight();
                }

                neuron.update_learning_rule(next_learning_rule_sum);
                next_learning_rule_sum = 0;
            }

        }
    }
}

void Perceptron::update_weights()
{
    for (std::vector<Neuron> &layer : this->neuron_layers)
    {
        for (Neuron &neuron : layer)
        {
            neuron.update_weights(this->learning_factor);
        }
    }
}