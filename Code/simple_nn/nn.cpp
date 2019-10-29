#include "nn.hpp"

 /* CIS631 Parallel processing - University of Oregon - Fall 2019
    Class project - Parallelizing Neural Networks
    Adam Noack & Trevor Bergstrom
    File Name: "nn.cpp"
    This file contains the class implementations of the neural network class, and node (neuron) class. All class and function definitions should be in the nn.hpp file
  */

/* creating a network should work as follows:
 * call the nework constructor
 *  specify the number of layers and the node count for each layer
 *
 * "Network" constructor.
 *      Creates the layers. Takes the node counts as an array and number of layers
 *      Then initialize layers and set prev and next layer pointers
 *
 *  Network connect
 *      will go through and connect all the nodes in the layers
 *
 *  Layer constructor
 *      Creates thge nodes in each layer
 *
 *  Connection constructor will create all the connection objects and store them in the respective node vectors
 *
 *  */



Network::Network(int* node_counts, int number_layers) {

    layers = new Layer*[num_layers];

    for(int i = 0; i < number_layers; i++) {
        Layer* new_layer = new Layer(node_counts[i]);
        layers[i] = new_layer;
    }

    for(int j = 0; j < number_layers - 1; j++) {
        layers[j]->add_connections(layers[j+1]);
    }
}

/* Layer constructor. Creates the layer container. */
Layer::Layer(int num_nodes) {

    previous_layer = NULL;
    next_layer = NULL;
    nodes = new Node*[num_nodes];

    for(int i = 0; i < num_nodes; i++) {
        Node* new_node = new Node(this);
        nodes[i] = new_node; //<-- Not sure here
    }
}

/* Node constructor */
Node::Node(Layer* a_layer) {
    num_output_connections = 0;
    num_input_connections = 0;
    activation = 0.0;
    cur_layer = a_layer;
}

/* Connection constructor */
Connection::Connection(Node* input, Node* output) {
    input_node = input;
    output_node = output;
    weight = 0.0;
}


/* This function adds connections between two layers, this layer and its next*/
void Layer::add_connections(Layer* next) {

    for(int i = 0; i < this->num_nodes; i++) {
        for(int j = 0; j < next->num_nodes; i++) {
            this->nodes[i]->create_connections(next->nodes[j]);
        }
    }
}

void Node::create_connections(Node* c_node) {
    Connection* c = new Connection(this, c_node);
    this->inputs.push_back(c);
    c_node->inputs.push_back(c);
}

Node::~Node() {
    for(int i = 0; i < outputs.size(); i++) {
        delete outputs[i];
    }
}

Layer::~Layer() {
    for(int i = 0; i < num_nodes; i++) {
        delete nodes[i];
    }
    delete nodes;
}

Network::~Network() {
    for(int i = 0; i < num_layers; i++) {
        delete layers[i];
    }
    delete layers;
}

Connection::~Connection(){}
