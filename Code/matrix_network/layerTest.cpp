#include "layer.hpp"

int main() {
    Layer* l1 = new Layer(2, input, NULL);
    Layer* l2 = new Layer(2, hidden, l1);
    Layer* l3 = new Layer(2, output, l2);

    float is[2] = {0.05, 0.1};
    float targets[2] = {0.01,  0.99};
    float w1[4] = {0.15, 0.2, 0.25, 0.3};
    float w2[4] = {0.4, 0.45, 0.50, 0.55};
    float b1[2] = { 0.35,0.35};
    float b2[2] = {0.6, 0.6};

    l1->set_output(is);
    l1->set_weights(w1);
    l2->set_weights(w2);
    l2->set_bias(b1);
    l3->set_bias(b2);

    printf("Layer 1\n");
    printf("Outputs:\n");
    l1->print_outputs();
    printf("Out weights\n");
    l1->print_out_weights();

    printf("Layer 2\n");
    printf("In weights\n");
    l2->print_in_weights();
    printf("inputs\n");
    l2->print_inputs();
    printf("Bias\n");
    l2->print_bias();
    printf("Out weights\n");
    l2->print_out_weights();

    printf("Layer 3\n");
    printf("In weights\n");
    l3->print_in_weights();
    printf("bias\n");
    l3->print_bias();

    printf("Forward l2\n");
    l2->forward_pass();
    printf("Forward l3\n");

    printf("l3 outs\n");
    l3->forward_pass();
    l3->print_outputs();

    float error = MSE(l3->outputs->get_row(0), targets, l3->outputs->num_cols);
    printf("Error: %f\n", error);

    printf("Backprop l3:\n");
    l3->back_prop(targets);
    //l3->print_del_bias();
    l3->print_in_del_W();

    printf("Backprop l2:\n");
    l2->back_prop(NULL);
    //l2->print_del_bias();
    l2->print_in_del_W();

    printf("----------------------- UPDATES ------------------------\n");

    l2->update(0.5, 1);
    printf("L2 bias\n");
    l2->print_bias();
    printf("l2 weights\n");
    l2->print_in_weights();
    printf("l3 update:\n");
    l3->update(0.5, 1);
    printf("L3 bias\n");
    l3->print_bias();
    printf("l3 weights\n");
    l3->print_in_weights();


}
