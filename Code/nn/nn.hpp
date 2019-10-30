#include <ctime>
#include <math.h>
#include <vector>
#include <cstdlib>

#define r_HI 10.0
#define r_LO 0.0
/* The network can be seen as a set of layers. Each layer contains a matrix of weights and a matrix of outputs */
class Layer;

class Network {
    private:
        int num_layers;
        Layer** layers;

    public:
        Network(int*, int);
        void connect();
        void forward_pass();
        void set_input(int*);
        void print_layers();
        void print_weights();
        ~Network();

};

class Layer {
    private:
        Layer* prev_layer;
        Layer* next_layer;
        int num_nodes;
        float** weights;
        float* outputs;
        float* bias;

    public:
        Layer(int);
        void set_output(int*);
        void set_next_layer(Layer* n) { next_layer = n; }
        void set_prev_layer(Layer* p) {prev_layer = p; }
        int get_num_nodes() { return num_nodes; }
        float* get_outputs() { return outputs; }
        float* get_bias() { return bias; }
        void connect_layers(Layer*);
        void compute_outputs(Layer*);
        void print_lweights();
        ~Layer();
};

float get_random_f();

float dot_prod(float*, float*, int);
