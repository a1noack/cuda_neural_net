#include "nn.hpp"
#include "utils/dataset.hpp"
#include <stdio.h>
#include <string>


int main() {
    char fname[] = "../data/data_n100_m5_mu1.5.csv";
    Dataset d(fname, 10);
    printf("%f", d.x[0][5]);
    printf("%f", d.minibatch[0][5]);
    
    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    int epochs = 5;

    Network* my_net = new Network(layout, num_layers);

    my_net->connect();
    for(int epoch = 1; epoch < epochs; epoch++) {
        for(int i = 0; i < d.n; i++) {
            my_net->set_input(d.x[i]);
            my_net->forward_pass();
            my_net->back_propogate(d.y[i]);
            my_net->update_weights();
        }
        printf("done with epoch %d\n", epoch);
    }
    delete my_net;
}

