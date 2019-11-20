#include "layers.hpp"

/************************************
 * Put CUDA kernels here
 ***********************************/





/***************************************
 * Class function implementations here
 ***************************************/

// Parent class

void Layer::connect(Layer* prev) {
    this->previous = prev;
    this->previous->set_next(this);
}

void Layer::set_next(Layer* nxt) {
    this->next = nxt;
}

void Layer::init_weights() {

    this->outputs = new matrix();
    this->bias = new matrix();
    this->dBias = new matrix();

    outputs->set_mem_zero(1,num_nodes);
    bias->set_mem_zero(1, num_nodes);
    dBias->set_mem_zero(1, num_nodes);

    int weight_count = this->next->get_num_nodes();
    weight_count *= this->num_nodes;

    weights = new matrix();
    dWeights = new matrix();
    weights-> set_mem_random(1, num_nodes);
    dWeights-> set_mem_random(1, num_nodes);
}

/*********************************************************
 * Linear Layer
 ********************************************************/

Linear_Layer::Linear_Layer(char* n_name, int nodes_num) {
    num_nodes = nodes_num;
    name = n_name;
}

/*********************************************************
 * RELU Layer
 ********************************************************/

RELU_Layer::RELU_Layer(char* n_name, int nodes_num) {
    num_nodes = nodes_num;
    name = n_name;
}

/*********************************************************
 * SIGMOID Layer
 ********************************************************/

Sigmoid_Layer::Sigmoid_Layer(char* n_name, int nodes_num) {
    num_nodes = nodes_num;
    name = n_name;
}



