#include "input.hpp"
#include <iostream>

Input::Input(Neuron &neuron) : neuron(neuron), weight((rand() / (double)RAND_MAX * 0.2) - 0.1){};

double Input::get_value()
{
    return this->neuron.get_value() * this->weight;
};

void Input::set_weight(double weight)
{
    this->weight = weight;
};

double Input::get_weight()
{
    return this->weight;
}

Neuron& Input::get_neuron()
{
    return this->neuron;
}