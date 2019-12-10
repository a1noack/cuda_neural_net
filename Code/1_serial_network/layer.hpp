#include "matrix.hpp"
#include <math.h>
#include <assert.h>

/* This is the layer class. It holds all of the parameters required for a neural network layer. These are stored in matrix data types. The layers implement the forward pass, back propogation, and update functions needed for network training. There are also some testing functions that should be left in. */

//enum to define a layer's position in a network
enum layer_pos {input, hidden, output};

class Layer {
    public:
        int num_nodes;
        layer_pos lp;

        Layer* prev;
        Layer* next;

        matrix* outputs;
        matrix* inputs;

        matrix* in_weights;
        matrix* out_weights;
        matrix* in_del_weights;
        matrix* out_del_weights;

        matrix* bias;
        matrix* del_bias;


        Layer(int, layer_pos, Layer*);
        ~Layer();

        void set_next_layer(Layer*);
        void set_output(float* ins) { outputs->set_memory(ins); }
        void zero_grad();

        void forward_pass();
        void back_prop(float* targets);
        void update(float, int);

        void print_layer();

        void set_weights(float* w) { out_weights->set_memory(w); }
        void set_bias(float* b) { bias->set_memory(b); }


        matrix* w_idx;
        matrix* n_idx;
};

float dot_prod(float**, float**, int);
float MSE(float*, float*, int);
