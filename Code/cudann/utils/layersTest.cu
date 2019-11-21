#include "layers.hpp"

#include <stdio.h>
#include <stdlib.h>

int main() {
    Linear_Layer* l1 = new Linear_Layer("Layer 1", 4);
    Linear_Layer* l2 = new Linear_Layer("Layer 2", 3);

    float l2w[12] = {1,2,3,4,2,3,4,5,3,4,5,6};
    float l1o[4] = {1,2,3,4};
    float l1b[4] = {0.0,0.0,0.0,0.0};
    float l2o[3] = {0,0,0};

    l2->set_weights(l2w, 3, 4);
    l1->set_outs(l1o, 4, 1);
    l1->set_bias(l1b, 4, 1);
    l2->set_outs(l2o, 3, 1);

    l1->print_outs();
    l2->print_weights();

    l2->connect(l1);
    printf("into forward\n");
    l2->forward();
}

