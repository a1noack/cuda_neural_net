#include "nn.hpp"

 /* CIS631 Parallel processing - University of Oregon - Fall 2019
    Class project - Parallelizing Neural Networks
    Adam Noack & Trevor Bergstrom
    File Name: "nn.cpp"
    This file contains the class implementations of the neural network class, and node (neuron) class. All class and function definitions should be in the nn.hpp file
  */

Network::Network(int* node_counts, int number_layers) {

    layers = new Layer*[num_layers];

    Layer* prev_layer = NULL;

    for(int i = 0; i < number_layers; i++) {
        Layer* new_layer = new Layer(node_counts[i]);
        if(prev_layer != NULL) {
            new_layer->previous_layer = prev_layer;
            prev_layer->next_layer = new_layer;
        }
    }


}

Layer::Layer(int num_nodes) {

    for(int i = 0; i < num_nodes; i++) {
        Node* new_node = new Node();
        nodes[i] = new_node();
    }

    //Adds the output connections for the nodes in the previous layer
    if(previous_layer != NULL) {
        for(int i = 0; i < prev_layer->num_nodes; i++) {
            for(int j = 0; j < num_nodes; j++) {
                prev_layer->nodes[i]->add_output_connection(this->nodes[j]);
            }
        }
    }


