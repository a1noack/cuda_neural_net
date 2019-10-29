#include "nn.hpp"

 /* CIS631 Parallel processing - University of Oregon - Fall 2019
    Class project - Parallelizing Neural Networks
    Adam Noack & Trevor Bergstrom
    File Name: "nn.cpp"
    This file contains the class implementations of the neural network class, and node (neuron) class. All class and function definitions should be in the nn.hpp file
  */

/* "Network" constructor. Creates the layers. Takes the node counts as an array and number of layers */
Network::Network(int* node_counts, int number_layers) {

    layers = new Layer*[num_layers];

       for(int i = 0; i < number_layers; i++) {
        Layer* new_layer = new Layer(node_counts[i]);
        layers[i] = new_layer;

        if(new_layer->prev_layer != NULL) {
            new_layer->previous_layer = prev_layer;
            prev_layer->next_layer = new_layer;
        }
    }
}

/* Layer constructor. Creates the layer container. */
Layer::Layer(int num_nodes) {

    previous_layer = NULL;
    next_layer = NULL;
    nodes = new Node*[num_nodes];

    for(int i = 0; i < num_nodes; i++) {
        Node* new_node = new Node();
        nodes[i] = new_node();
    }

}

/* This function adds connections between the current layer and the previous layer */
Layer::add_connections() {

    if(previous_layer != NULL) {
        for(int i = 0; i < prev_layer->num_nodes; i++) {
            for(int j = 0; j < num_nodes; j++) {
                prev_layer->nodes[i]->add_output_connection(this->nodes[j]);
                this->nodes[j]->add_input_connection(prev_layer->nodes[i]);
            }
        }
    }
}

Node::Node(Layer* a_layer) {
    num_output_connections = 0;
    num_input_connections = 0;
    activation = 0.0;
    cur_layer = a_layer;

    inputs = new connection_list;
    inputs->connections = new Connection*[];

}

Node::add_output_connection(Node* next) {
    Connection* n_connection = new Connection(this, next);
    outputs[out_itr] = n_connection;
}

Connection::Connection(Node* input, Node* output) {
    input_node = input;
    output_node = output;
    weight = 0.0;
}


