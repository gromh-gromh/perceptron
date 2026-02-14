.PHONY: all debug build

all: build ./build/main.out
	./build/main.out

debug: build ./build/main.out
	gdb -q -ex run ./build/main.out

build: ./src/perceptron/neuron/neuron.hpp ./src/perceptron/neuron/neuron.cpp ./src/perceptron/input/input.hpp ./src/perceptron/input/input.cpp ./src/perceptron/perceptron.hpp ./src/perceptron/perceptron.cpp ./src/main.cpp
	g++ -lstdc++ -std=c++20 -o ./build/main.out ./src/perceptron/neuron/neuron.hpp ./src/perceptron/neuron/neuron.cpp ./src/perceptron/input/input.hpp ./src/perceptron/input/input.cpp ./src/perceptron/perceptron.hpp ./src/perceptron/perceptron.cpp ./src/main.cpp -lm -g