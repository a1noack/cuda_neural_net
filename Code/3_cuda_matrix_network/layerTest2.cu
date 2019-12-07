#include "layer.hpp"

int main() {
    Layer* l1 = new Layer(2, input, NULL, 1);
    Layer* l2 = new Layer(2, hidden, l1, 1);
    Layer* l3 = new Layer(2, output, l2, 1);

    float is[2] = {0.05, 0.1};
    float targets[2] = {0.01,  0.99};
    matrix* tar = new matrix(1, 2);
    tar->move_to_device();
    matrix* temp = new matrix(1,2);
    temp->move_to_device();

    float w1[2] = {0.15, 0.2};
    float w2[2] = {0.25, 0.3};

    //float w_l1[4] = {0.15, 0.2, 0.25, 0.3};
    float w_l1[4] = {0.15, 0.25, 0.2, 0.3};

    float w6[2] = {0.4, 0.45};
    float w7[2] = {0.5, 0.55};

    //float w_l2[4] = {0.4, 0.45, 0.5, 0.55};
    float w_l2[4] = {0.4, 0.5, 0.45, 0.55};

    float b_l2[3] = {0.35,0.35};
    float b_l3[2] = {0.6, 0.6};

    l1->set_weights(w_l1);
    l2->set_weights(w_l2);
    l2->set_bias(b_l2);
    l3->set_bias(b_l3);

    l2->move_to_device();
    l3->move_to_device();


    float error = 1.0;
    int j = 0;
    while(j < 1) {
        j++;
        l1->zero_grad();
        l2->zero_grad();
        l3->zero_grad();

        l1->set_output(is);
        tar->set_memory(targets);

        /*
        printf("Layer 1:\n");
        l1->print_layer();
        printf("Layer 2:\n");
        l2->print_layer();
        printf("Layer 3:\n");
        l3->print_layer();
*/

        l2->forward_pass();
        l3->forward_pass();
        error = MSE_mat_wrapper(l3->outputs, tar, temp);
        printf("Error for sample 1: %f\n", error);

        l3->back_prop(tar, 1);
        l2->back_prop(NULL, 1);

        l3->update(0.5, 1);
        l2->update(0.5, 1);
        printf("Layer 1:\n");
        l1->print_layer();
        printf("Layer 2:\n");
        l2->print_layer();
        printf("Layer 3:\n");
        l3->print_layer();

}

}
