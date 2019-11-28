#include "nn.hpp"
#include <stdio.h>

int main() {
    int layout[3] = {2,2,2};
    int num_layers = 3;

    layer_type lts[3] = {Sigmoid, Sigmoid, Sigmoid};

    Network* my_net = new Network(layout, num_layers, lts);

    my_net->connect();

    float l3bias[2] = {0.6, 0.6};
    float l2bias[2] = {0.35, 0.35};

    float l3w[4] = {0.4, 0.45, 0.5, 0.55};
    float l2w[4] = {0.15, 0.2, 0.25, 0.3};

    float targets[2] = {0.01, 0.99};
    float ins[2] = {0.05, 0.1};

    float* weights[2] = {l2w, l3w};
    float* bias[2] = {l2bias, l3bias};

    my_net->zero_grad();
    printf("set w\n");
    my_net->set_weights(weights);
    printf("set b\n");
    my_net->set_bias(bias);

    my_net->set_input(ins);

    printf("weights before FWp\n");
    my_net->print_weights();
    printf("bias before FWp\n");
    my_net->print_bias();

    printf("FWp start\n");
    my_net->forward_pass();

    printf("FWp over\n");
    float* result = my_net->get_output();

    printf("REsults: %f, %f\n", result[0], result[1]);

    float err = MSE(my_net->get_output(), targets, 2);
    printf("\nError: %f\n", err);
    printf("Start back prop\n");

    my_net->back_propogate(targets);

    printf("end backprop\n");

    printf("start updates\n");
    my_net->update_weights(0.5, 1);
    printf("start updates\n");

    printf("weights after FWp\n");
    my_net->print_weights();
    printf("bias after FWp\n");
    my_net->print_bias();


}

