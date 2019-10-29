#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

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
        float bias;
        float output_value;
        /* other stuff the neuron needs */

    public:
        Node(Layer*); //Constructor
        void create_connections(Node*);
        void set_bias(float b) { bias = b; }
        void compute_activation();
        float get_bias() {return bias;}
        float get_output() {return output_value;}
        void set_output(float o) {output_value = o;}
        int get_output_count() { return outputs.size(); }

        Connection* get_out_connection_at(int i) { return outputs[i]; }
        ~Node();
};

class Connection {
    private:
        Node* input_node;
        Node* output_node;
        float weight;

    public:
        Connection(Node*, Node*);
        Node* get_input() {return input_node;}
        Node* get_output() {return output_node;}
        void set_weight(float w) {weight = w; printf("weight set\n");}
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
        int get_num_nodes() { return num_nodes; }
        Layer(int num_nodes);
        Node* get_node_at(int i) { return nodes[i]; }
        void add_connections(Layer*);
        ~Layer();
};

class Network {
    private:
        int num_layers;
        Layer** layers; //List of layers in the network

    public:
        Network(int* node_counts, int number_layers);
        void randomize_weights();
        void randomize_bias();
        ~Network();
};
