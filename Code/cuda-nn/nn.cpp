#include "nn.hpp"
#include <stdio.h>

/* Network Constructor - creates the network class you taking a list of the node counts in each layer
 * and number of layers.*/
Network::Network(int* numNodes, int numLayers, layer_type* types) {

    num_layers = numLayers;

    layers = new Layer*[num_layers];

    for(int i = 0; i < num_layers; i++) {
        Layer* nl = NULL;
        switch(types[i])
        {
            case(RELU): nl = new RELU_Layer(numNodes[i]); break;
            case(Sigmoid): nl = new Sigmoid_Layer(numNodes[i]); break;
            case(Softmax): nl = new Softmax_Layer(numNodes[i]); break;
        }
        //Layer* nl = new Sigmoid_Layer(numNodes[i]);
        layers[i] = nl;
    }
    //printf("Network Created\n");
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
    //printf("Network Connected\n");
}

/* Preform a forward pass on the network. Computes outputs for each layer */
void Network::forward_pass() {
    for(int i = 1; i < num_layers; i++) {
        //printf("Forward Pass for Layer %d\n", i);
        layers[i]->forward_pass(layers[i-1]);
    }
}

/* Stupid. Sets the input data on the first layer */
void Network::set_input(float* data) {
    layers[0]->set_output(data);
}


float* Network::get_output() {
    return this->layers[this->num_layers - 1]->get_outputs()->host_data;
}

/* testing function, prints the outputs of each node */
void Network::print_layers() {
    for(int i = 0; i < num_layers; i++) {
        //printf("For layer #%d, with %d nodes: ", i, layers[i]->get_num_nodes());
        float* od = layers[i]->get_outputs()->host_data;
        for(int j = 0; j < layers[i]->get_num_nodes(); j++) {
            printf("%f ", od[j]);
        }
        //printf("\n");
    }
}



/* Function that backpropogates error thru the entire network */
void Network::back_propogate(float* targets) {
    int i = num_layers - 1;
    //printf("backprop last layer\n");
    layers[i]->back_prop_input(targets);
    i--;

    for(; i >=1; i--) { // Not bp for the input layer
        //printf("back prop layer %d\n", i);
        layers[i]->back_prop();
    }
}



/* update network weights */
void Network::update_weights(float learn_rate, int batch_size) {
    for(int i = num_layers -1; i >=1; i--) {
        layers[i]->update(learn_rate, batch_size);
    }
}

void Network::zero_grad() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->zero_grad();
    }
}

/* testing function prints weights of each connection */
void Network::print_weights() {
    for(int i = 1; i < num_layers; i++) {
        layers[i]->print_lweights();
    }
}

void Network::set_weights(float** a) {
    for(int i = 1; i <= num_layers; i++) {
        //printf("setting weights for layer %d\n", i);
        layers[i]->set_weights(a[i-1]);
    }
}

void Network::set_bias(float** a) {
    for(int i = 1; i <= num_layers; i++) {
        layers[i]->set_bias(a[i-1]);
    }
}

void Network::print_bias() {
    for(int i = 1; i <= num_layers; i++) {
        layers[i]->print_bias();
    }
}
