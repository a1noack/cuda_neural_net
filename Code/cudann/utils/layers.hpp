/* this is the implementation for the layer class. There should be a virtual parent class and then the different types of layers should be created form there.
 *
 * Each layer defines its own forward and backward pass using its own activation function.*/

#include "matrix.hpp"
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREADS_PER_BLOCK 512

class Layer {
    protected:
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
        void set_next(Layer*);
        void init_weights();
        virtual void forward() = 0;
        virtual void backprop() = 0;
        int get_num_nodes() { return num_nodes;}
        matrix* get_outputs() { return outputs; }

        //Testing functions!!
        void set_weights(float*, int, int);
        void set_bias(float*, int, int);
        void set_outs(float*, int, int);
        void print_outs();
        void print_weights();
        void print_bias();
};

class Linear_Layer: public Layer{
    private:


    public:
        Linear_Layer(char*, int);
        ~Linear_Layer();
        void forward();
        void backprop();
};

class RELU_Layer: public Layer {
    private:

    public:
        RELU_Layer(char*, int);
        ~RELU_Layer();
        void forward();
        void backprop();
};

class Sigmoid_Layer: public Layer {
    private:

    public:
        Sigmoid_Layer(char*, int);
        ~Sigmoid_Layer();
        void forward();
        void backprop();
};


