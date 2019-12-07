#include "network.hpp"

int main() {
    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    layer_type lts[4] = {Sigmoid, Sigmoid, Sigmoid, Sigmoid};

    Network* my_net = new Network(num_layers, layout, lts);

    int epochs = 100;
    int batch_sz = 1;
    float learn_r = 0.01;
    float min_error = 0.001;
    my_net->train(epochs, batch_sz, learn_r, min_error);


}
