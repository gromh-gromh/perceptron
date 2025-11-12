#pragma once
#include <stddef.h>
#include <vector>
#include <cstdlib>
#include "../neuron/neuron.hpp"

class Neuron;

class Input{
    public:
        Input(Neuron &neuron);
        double get_value();
        void set_weight(double weight);
        double get_weight();
        Neuron& get_neuron();

    private:
        Neuron &neuron;
        double weight;
};