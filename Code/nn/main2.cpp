#include "nn.hpp"
#include "../data/data_loader.hpp"
#include <stdio.h>
#include <string>


int main() {
    std::string fname = "..data/data_n100_m5_mu1.5.csv";
    float **x, **y;
    int *n, *m;
    load_data(fname, x, y, n, m);

    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    int epochs = 5;

    Network* my_net = new Network(layout, num_layers);

    my_net->connect();
    
    for(int epoch = 1; epoch < epochs; epoch++) {
        for(int i = 0; i < *n; i++) {
            my_net->set_input(x[i]);
            my_net->forward_pass();
            my_net->back_propogate(y[i]);
            my_net->update_weights();
        }
    }
    delete my_net;
}

