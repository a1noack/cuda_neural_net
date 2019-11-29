#include "network.hpp"

Network::Network(int num_l, int* node_list, layer_type* lt) {
    num_layers = num_l;

    layers = new Layer*[num_layers];

    Layer* last_layer = NULL;

    for(int i = 0; i < num_layers; i++) {
        Layer* nl = NULL;

        layer_pos position = hidden;

        if(i == 0) {
            position = input;
        } else if (i == num_layers - 1) {
            position = output;
        }

        switch(lt[i]) {
            case(RELU): nl = new Layer(node_list[i], position, last_layer); break;
            case(Sigmoid): nl = new Layer(node_list[i], position, last_layer); break;
            case(Softmax): nl = new Layer(node_list[i], position, last_layer); break;
        }

        last_layer = nl;
        layers[i] = nl;
    }
}

void Network::train(int num_epocs, int batch_size, ) {

