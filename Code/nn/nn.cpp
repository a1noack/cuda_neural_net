#include "nn.hpp"
#include <stdio.h>

/* Network Constructor - creates the network class you taking a list of the node counts in each layer
 * and number of layers.*/
Network::Network(int* numNodes, int numLayers) {

    num_layers = numLayers;

    layers = new Layer*[num_layers];

    for(int i = 0; i < num_layers; i++) {
        Layer* nl = new Layer(numNodes[i]);
        layers[i] = nl;
    }
    printf("Network Created\n");
}

/* Clean up a network */
Network::~Network() {
    for(int i = 0; i < num_layers; i++) {
        delete layers[i];
    }
    delete layers;
}

/* Connect layers in the network (Fully Connected Network) */
void Network::connect() {
    for(int i = 1; i < num_layers; i++) {
        printf("attempting to connect layers %d and %d\n", layers[i]->get_num_nodes(), layers[i-1]->get_num_nodes());
        layers[i]->connect_layers(layers[i-1]);
    }
    printf("Network Connected\n");
}

/* Preform a forward pass on the network. Computes outputs for each layer */
void Network::forward_pass() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->compute_outputs(layers[i-1]);
    }
}

/* Stupid. Sets the input data on the first layer */
void Network::set_input(float* data) {
    layers[0]->set_output(data);
}

/* Again, stupid. Sets all the outputs of the first (input) layer, as the training data example. */
void Layer::set_output(float* data) {
    for(int i = 0; i < num_nodes; i++) {
        outputs[i] = data[i];
    }
}

/* testing function, prints the outputs of each node */
void Network::print_layers() {
    for(int i = 0; i < num_layers; i++) {
        printf("For layer #%d, with %d nodes: ", i, layers[i]->get_num_nodes());
        float* od = layers[i]->get_outputs();
        for(int j = 0; j < layers[i]->get_num_nodes(); j++) {
            printf("%f ", od[j]);
        }
        printf("\n");
    }
}

/* testing function prints weights of each connection */
void Network::print_weights() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->print_lweights();
    }
}

/* testing function, prints each weight */
void Layer::print_lweights() {
    for(int i = 0; i <num_nodes; i++) {
        float* f = weights[i];
        for(int j = 0; j < prev_layer->get_num_nodes(); j++) {
            printf("%f ", f[j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Layer constructor. Initializes many data members. What it cannot it will set to NULL, hopefully avoid shitty behavior */
Layer::Layer(int numNodes) {
    this->num_nodes = numNodes;
    prev_layer = NULL;
    next_layer = NULL;

    weights = NULL;
    outputs = new float[numNodes];

    bias = new float[numNodes];
    del_bias = new float[numNodes];

    for(int i = 0; i < numNodes; i++) {
        bias[i] = get_random_f();
        outputs[i] = 0.0;
        del_bias[i] = 0.0;
    }

    printf("layer Created %d nodes\n", this->num_nodes);
}

/* SIGMOID ACTIVATION!!! */
void Layer::compute_outputs(Layer* prev) {
    for(int i = 0; i < num_nodes; i++) {
        float dp = dot_prod(weights[i], prev->get_outputs(), num_nodes);
        outputs[i] = (1 / (1+ expf(-1*dp + bias[i])));
    }
}

/* Clearly a incomplete destructor */
Layer::~Layer() {
}

/* function to connect layers with weights and randomize weights, each layer owns its input edges! */
void Layer::connect_layers(Layer* prev) {
    prev->set_next_layer(this);
    this->set_prev_layer(prev);
    this->weights = new float*[num_nodes];
    this->del_weights = new float*[num_nodes];

    for(int i = 0; i < this->num_nodes; i++) {
        float* wl = new float[prev->get_num_nodes()];

        for (int j = 0; j < prev->get_num_nodes(); j++) {
            wl[j] = get_random_f();
        }

        weights[i] = wl;
    }
    printf("Layer Connected\n");
}

/* Get a random float */
float get_random_f() {

    return r_LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(r_HI-r_LO)));
}

/* Calc dot product of two arrays of floats */
float dot_prod(float* x, float* y, int num) {

    float dp = 0.0;
    for(int i = 0; i < num; i++) {
        dp += x[i] * y[i];
    }

    return dp;
}

/* calculate mean squared error of two float arrays */
float MSE(float* v1, float* v2, int n) {
    float *s = new float[n];
    for(int i = 0; i < n; i++) {
        s[i] = pow(static_cast <double> (v1[i] - v2[i]), 2);
    }

    float len = s[0];
    for(int j = 1; j < n; j++) {
        len -= s[j];
    }

    return (1/n) * len;
}

/* Function that backpropogates error thru the entire network */
void Network::back_propogate(float* targets) {
    int i = num_layers - 1;
    layers[i]->back_prop_input(targets);
    i--;

    for(; i >=1; i--) { // Not bp for the input layer
        printf("back prop layer %d\n", i);
        layers[i]->back_prop();
    }
}

/* The good shit. Back propogate the error on just the input layer. */
void Layer::back_prop_input(float* targets) {
    int num_weights = prev_layer->get_num_nodes();

    for(int i = 0; i < num_nodes; i++) {
        float o = outputs[i];
        del_bias[i] = (o - targets[i]) * (o * (1 - o));
        float* dw = new float[num_weights];
        float* po = prev_layer->get_outputs();

        for(int j = 0; j < num_weights; j++) {
            dw[j] = del_bias[i] * po[i];
        }

        del_weights[i] = dw;
    }

}

/* the really good shit. Back propgates the error for the hidden layers */
void Layer::back_prop() {
    int num_weights = prev_layer->get_num_nodes();
    float* ndb = next_layer->get_del_bias();

    for(int i = 0; i < num_nodes; i++) {
        float** w = next_layer->get_weights();

        int next_num_weights = next_layer->get_num_nodes();
        float db = 0.0;

        //printf("Backprop node %d\n", i);

        for(int j = 0; j < next_layer->get_num_nodes(); j++) {
            //printf("summing del_bias itr %d\n", j);
            //float x = ndb[j];
            //printf("------------\n");
            //float y = w[j][i];
            //printf("summing del_bias itr %d ndb = %f, w = %f\n", j, x, y);
            db += ndb[j] * w[j][i];
        }

        //printf("done calc del bias for node %d\n", i);
        float o = outputs[i];
        del_bias[i] = db * (o * (1-o));

        float* dw = new float[num_nodes];

        float* po = prev_layer->get_outputs();

        for(int k = 0; k < num_nodes; k++) {
            dw[k] = del_bias[i] * po[k];
        }

        del_weights[i] = dw;
    }
}

/* update network weights */
void Network::update_weights() {
    for(int i = num_layers -1; i >=1; i--) {
        layers[i]->update();
    }
}

/* update weights on each layer */
void Layer::update() {
    double learn_rate = 0.5; //<---------------- Maybe put this somewhere else

    int num_weights = prev_layer->get_num_nodes(); //<------- BAD

    for(int i = 0; i < num_nodes; i++) {
        float* w = weights[i];
        float* dw = del_weights[i];

        for(int j = 0; j < num_weights; j++) {
            w[j] = w[j] - (learn_rate * dw[j]);
        }

        bias[i] = bias[i] - (learn_rate - del_bias[i]);
    }
}

