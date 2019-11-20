/* this is the implementation for the layer class. There should be a virtual parent class and then the different types of layers should be created form there.
 *
 * Each layer defines its own forward and backward pass using its own activation function.*/

#include "matrix.hpp"
#include <stdio.h>
#include <stdlib.h>

class Layer {
    private:
        Layer* next;
        Layer* previous;

        char* name;
        int num_nodes;

        matrix* weights;
        matrix* dWeights;
        matrix* outputs;
        matrix* bias;
        matrix* dBias;

    public:
        Layer() {};
        ~Layer() {};
        void connect(Layer*);
        void init_weights();
        virtual void forward() = 0;
        virtual void backprop() = 0;
};

class Linear_Layer:public Layer{
    private:


    public:
        Linear_Layer(char*, int);
        ~Linear_Layer();
        void forward();
        void backprop();
};

class RELU_Layer:public Layer {
    private:

    public:
        RELU_Layer(char*, int);
        ~RELU_Layer();
        void forward();
        void backward();
};

class Sigmoid_Layer:public Layer {
    private:

    public:
        Sigmoid_Layer(char*, int);
        ~Sigmoid_Layer();
        void forward();
        void backward();
};


