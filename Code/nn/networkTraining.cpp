#include "nn.hpp"
#include "utils/dataset.hpp"
#include <stdio.h>

int main(int argv, char** argc) {

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";

    Dataset d(file_name, 10);

    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    int epochs = 10;

    Network* my_net = new Network(layout, num_layers);

    my_net->connect();

    for(int i = 0; i < epochs; i++) {
        d.shuffle_sample_order();

        for(int j = 0; j < something; j++) {
        }
    }
}






