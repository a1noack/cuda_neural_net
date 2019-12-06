#include "network.hpp"

int main() {
    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    layer_type lts[4] = {Sigmoid, Sigmoid, Sigmoid, Sigmoid};\

    int epochs = 1000;
    int batch_sz = 1;
    float learn_r = 0.001;
    float min_error = 0.001;

    Network* my_net = new Network(num_layers, layout, lts, batch_sz);

    //printf("Start Training!\n");
    my_net->train(epochs, batch_sz, learn_r, min_error);
}
