#include "nn.hpp"

int main2() {
    srand(static_cast <unsigned> (time(NULL)));
    int layout[3] = {2,2,2};
    int n = 3;

    float in_dat[2] = {0.05,0.1};

    float w1[4] = {0.15, 0.2, 0.25, 0.3};
    float w2[4] = {0.4, 0.45, 0.5, 0.55};
    float* wt[2] = {w1,w2};

    float b1[2] = {0.35, 0.35};
    float b2[2] = {0.6, 0.6};
    float* bt[2] = {b1,b2};

    float target[2] = {0.01, 0.99};

    Network* my_net = new Network(layout, n);

    my_net->connect();

    my_net->set_input(in_dat);

    my_net->set_weights(wt);
    my_net->set_bias(bt);

    printf("printing layers: \n");
    my_net->print_layers();

    printf("printing weights: \n");
    my_net->print_weights();

    printf("Printing Bias: \n");
    my_net->print_bias();

    printf("forward pass\n");
    my_net->forward_pass();

    printf("\n");
    float err = MSE(my_net->get_output(), target, 2);
    printf("MSE: %f\n", err);
    printf("\n");

    printf("printing layers: \n");
    my_net->print_layers();

    printf("printing weights: \n");
    my_net->print_weights();
    printf("Printing Bias: \n");
    my_net->print_bias();

    printf("\n\n");

    float out_dat[2] = {0,1};

    my_net->back_propogate(target);
    my_net->update_weights();

    printf("printing weights: \n");
    my_net->print_weights();
    printf("Printing Bias: \n");
    my_net->print_bias();

    delete my_net;
}
