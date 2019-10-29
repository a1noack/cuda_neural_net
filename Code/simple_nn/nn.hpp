#include <iostream>
#include <vector>

/* CIS631 Parallel processing -Univeristy of Oregon - Fall 2019
   Class project - Parallelizing Neural Networks
   Adam Noack & Trevor Bergstrom
   File Name: "nn.hpp"
   This file contains the class and function definitions of the neural network class, and node (neuron) class. All class and      function implementations should be in the nn.cpp file
   */

class Connection;
class Layer;

class Node {
    private:
        int num_output_connections;
        int num_input_connections;

        std::vector<Connection*> inputs; //<- Create dynamic array class? Or just use vector?
        std::vector<Connection*> outputs;

        //Connection** inputs; //list of input connections
        //Connection** outputs; //list of output connections
        Layer* cur_layer; //layer that node exists in

        float activation;
        /* other stuff the neuron needs */

    public:
        Node(Layer*); //Constructor
        void create_connections(Node*);
        void compute_activation();
        //float get_activation();
        ~Node();
};

class Connection {
    private:
        Node* input_node;
        Node* output_node;
        float weight;

    public:
        Connection(Node*, Node*);
        float get_weight() {return weight;}
        ~Connection();
};

class Layer {
    private:
        int num_nodes;
        Node** nodes; //Nodes in current layer
        Layer* previous_layer;
        Layer* next_layer;

    public:
        Layer(int num_nodes);
        void add_connections(Layer*);
        ~Layer();
};

class Network {
    private:
        int num_layers;
        Layer** layers; //List of layers in the network

    public:
        Network(int* node_counts, int number_layers);
        ~Network();
};
