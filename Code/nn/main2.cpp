#include "nn.hpp"
#include "../data/data_loader.hpp"
#include <stdio.h>
#include <string>


int main() {
    std::string fname = "../data/data_n100_m5_mu1.5.csv";
    float **x, **y;
    int args[3];
    int n, m, k;
    load_data(fname, x, y, args);
    n = args[0];
    m = args[1];
    k = args[2];

    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    int epochs = 5;

    Network* my_net = new Network(layout, num_layers);

    my_net->connect();
    printf("x[1][5] = %f", x[1][5]);
    printf("n = %d \n", n); 
    for(int epoch = 1; epoch < epochs; epoch++) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) printf("%f ", x[i][j]);
            my_net->set_input(x[i]);
            my_net->forward_pass();
            my_net->back_propogate(y[i]);
            my_net->update_weights();
        }
    }
    delete my_net;
}

