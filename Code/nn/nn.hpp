#include <ctime>
#include <math.h>
#include <vector>
#include <cstdlib>

#define r_HI 1.0
#define r_LO 0.0
/* The network can be seen as a set of layers. Each layer contains a matrix of weights and a matrix of outputs */

/* --------------------------_TO DO----------------------------------------------------------
 * Could get rid of the private variables... Creating many un-needed functions
 * Need to create a data loader. That can define the number of output and input nodes.
 * May need to rewrite input and output layers as derived classes?
 * Sleep.
 * Destructor for layers!!!! Clean up memory leaks. Probably do this before scaling.
 * --------------------------------------------------------------------------------------- */

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

        void back_propogate(int*);
        void update_weights();

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

        float** del_weights;
        float* del_bias;

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

        void back_prop_input(int*);
        void back_prop();

        float** get_weights() { return weights; }
        float* get_del_bias() { return del_bias; }
        float** get_del_weights() { return del_weights; }

        void update();
        ~Layer();
};

float get_random_f();

float dot_prod(float*, float*, int);

float MSE(float*, float*, int);
