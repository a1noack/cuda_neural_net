#include <iostream>

/* CIS631 Parallel processing -Univeristy of Oregon - Fall 2019
   Class project - Parallelizing Neural Networks
   Adam Noack & Trevor Bergstrom
   File Name: "nn.hpp"
   This file contains the class and function definitions of the neural network class, and node (neuron) class. All class and      function implementations should be in the nn.cpp file
   */

typedef struct {
    Connection** connections;
    int itr;
} connection_list;

class Node {
    private:
        int num_output_connections;
        int num_input_connections;

        int in_itr;
        int out_itr;

        connection_list* inputs;
        connection_list* outputs;

        //Connection** inputs; //list of input connections
        //Connection** outputs; //list of output connections
        Layer* cur_layer; //layer that node exists in

        float activation;
        /* other stuff the neuron needs */

    public:
        Node(); //Constructor
        void add_output_connection(Node* next_node);
        void add_input_connection(Node* prev_node);
        void compute_activation();
        float get_activation();
        ~Node();
};

class Connection {
    private:
        Node* input_node;
        Node* output_node;
        float weight;

    public:
        Connection(Node* input, Node* output);
        float get_weight() {return weight;}
        ~Connection();
};

class Layer {
    private:
        int num_nodes
        Node** nodes; //Nodes in current layer
        Layer* previous_layer;
        Layer* next_layer;

    public:
        Layer(int num_nodes);
        void initialize_nodes(int num_nodes);
        ~Layer();
};

class Network {
    private:
        Layer** layers; //List of layers in the network

    public:
        Network(int* node_counts, int number_layers);
        ~Network();
};
