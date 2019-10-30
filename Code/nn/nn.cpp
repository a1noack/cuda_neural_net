#include "nn.hpp"

Network::Network(int* numNodes, int numLayers) {

    num_layers = numLayers;

    layers = new Layer*[num_layers];

    for(int i = 0; i < num_layers; i++) {
        Layer* nl = new Layer(numNodes[i]);
        layers[i] = nl;
    }
    printf("Network Created\n");
}

Network::~Network() {
    for(int i = 0; i < num_layers; i++) {
        delete layers[i];
    }
    delete layers;
}

void Network::connect() {
    for(int i = 1; i < num_layers; i++) {
        printf("attempting to connect layers %d and %d\n", layers[i]->get_num_nodes(), layers[i-1]->get_num_nodes());
        layers[i]->connect_layers(layers[i-1]);
    }
    printf("Network Connected\n");
}

void Network::forward_pass() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->compute_outputs(layers[i-1]);
    }
}

void Network::set_input(int* data) {
    layers[0]->set_output(data);
}

void Layer::set_output(int* data) {
    for(int i = 0; i < num_nodes; i++) {
        outputs[i] = data[i];
    }
}

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

void Network::print_weights() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->print_lweights();
    }
}

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

Layer::Layer(int numNodes) {
    this->num_nodes = numNodes;
    prev_layer = NULL;
    next_layer = NULL;

    weights = NULL;
    outputs = new float[numNodes];

    bias = new float[numNodes];
    for(int i = 0; i < numNodes; i++) {
        bias[i] = get_random_f();
        outputs[i] = 0.0;
    }

    printf("layer Created %d nodes\n", this->num_nodes);
}

void Layer::compute_outputs(Layer* prev) {
    for(int i = 0; i < num_nodes; i++) {
        float dp = dot_prod(weights[i], prev->get_outputs(), num_nodes);
        outputs[i] = 1/(1+ expf(-1*dp));
    }
}

Layer::~Layer() {
}

void Layer::connect_layers(Layer* prev) {
    prev->set_next_layer(this);
    this->set_prev_layer(prev);
    this->weights = new float*[num_nodes];
    for(int i = 0; i < this->num_nodes; i++) {
        float* wl = new float[prev->get_num_nodes()];

        for (int j = 0; j < prev->get_num_nodes(); j++) {
            wl[j] = get_random_f();
        }

        weights[i] = wl;
    }
    printf("Layer Connected\n");
}

float get_random_f() {

    //srand (static_cast <unsigned> (time(NULL)));
    return r_LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(r_HI-r_LO)));
}

float dot_prod(float* x, float* y, int num) {

    float dp = 0.0;
    for(int i = 0; i < num; i++) {
        dp += x[i] * y[i];
    }

    return dp;
}





