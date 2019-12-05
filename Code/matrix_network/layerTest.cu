#include "layer.hpp"

int main() {
    Layer* l1 = new Layer(5, input, NULL, 1);
    Layer* l2 = new Layer(3, hidden, l1, 1);
    Layer* l3 = new Layer(2, output, l2, 1);

    float is[5] = {0.05, 0.1, 0.35, 0.75, 0.25};
    float targets[2] = {0.01,  0.99};
    matrix *tar = new matrix(1, 2);
    tar->set_memory(targets);
    tar->move_to_device();

    float w1[3] = {0.15, 0.2, 0.25};
    float w2[3] = {0.4, 0.45, 0.50};
    float w3[3] = {0.3, 0.35, 0.4};
    float w4[3] = {0.25, 0.2, 0.15};
    float w5[3] = {0.5, 0.45, 0.4};

    float w_l1[15] = {};
    int i = 0;
    for(int j = 0; j < 3; j++) {
        w_l1[i] = w1[j];
        i++;
    }
    for(int j = 0; j < 3; j++) {
        w_l1[i] = w2[j];
        i++;
    }
    for(int j = 0; j < 3; j++) {
        w_l1[i] = w3[j];
        i++;
    }
    for(int j = 0; j < 3; j++) {
        w_l1[i] = w4[j];
        i++;
    }
    for(int j = 0; j < 3; j++) {
        w_l1[i] = w5[j];
        i++;
    }

    float w_l2[6] = {};

    float w6[2] = {0.75, 0.6};
    float w7[2] = {0.15, 0.1};
    float w8[2] = {0.35, 0.65};

    w_l2 [0] = w6[0];
    w_l2 [1] = w6[1];
    w_l2 [2] = w7[0];
    w_l2 [3] = w7[1];
    w_l2 [4] = w8[0];
    w_l2 [5] = w8[1];

    float b_l2[3] = {0.35,0.35, 0.35};
    float b_l3[2] = {0.6, 0.6};


    l1->set_output(is);
    l1->set_weights(w_l1);
    l2->set_weights(w_l2);
    l2->set_bias(b_l2);
    l3->set_bias(b_l3);

    printf("Layer1\n");
    l1->print_layer();
    printf("Layer2\n");
    l2->print_layer();
    printf("Layer3\n");
    l3->print_layer();

    l2->move_to_device();
    l3->move_to_device();

    printf("L2 forward\n");
    l2->forward_pass();

    printf("L3 forward\n");
    l3->forward_pass();

    l2->move_to_host();
    l3->move_to_host();

    printf("-----------NETWORK OUTPUT ------------------)\n\t");
    l3->outputs->print();
    printf("--------------------------------------------)\n");

    float error = MSE(l3->outputs->get_row(0), targets, l3->outputs->num_cols);
    printf("-----------NETWORK ERROR -------------------)\n");
    printf("\t%f\n", error);
    printf("--------------------------------------------)\n");

    printf("<-------------------Back prop L3: ----------------->\n");
    //l3->print_layer();
    l3->back_prop(tar, 1);
    //l3->print_layer();
    printf("<-------------------------------------------------->\n");

    printf("<-------------------Back prop L2: ----------------->\n");
    //l2->print_layer();
    l2->back_prop(NULL, 1);
    //l2->print_layer();
    printf("<-------------------------------------------------->\n");

    printf("<-----------------------Update L3 ----------------->\n");
    //l3->print_layer();
    l3->update(0.5, 1);
    //l3->print_layer();
    printf("<-------------------------------------------------->\n");


    printf("<-----------------------Update L2 ----------------->\n");
    //l2->print_layer();
    l2->update(0.5, 1);
    //l2->print_layer();
    printf("<-------------------------------------------------->\n");

    float ins2[5] = {0.85, 0.3, 0.15, 0.9, 0.45};
    float targets2[2] = {0.01, 0.99};

    //l1->zero_grad();
    //l2->zero_grad();
    //l3->zero_grad();
    //l1->set_output(ins2);
    //l2->forward_pass();
    //l3->forward_pass();

    error = MSE(l3->outputs->get_row(0), targets2, l3->outputs->num_cols);
    printf("-----------NETWORK ERROR -------------------)\n");
    printf("\t%f\n", error);
    printf("--------------------------------------------)\n");

        /*
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
*/
}
