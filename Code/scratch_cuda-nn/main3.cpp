#include "nn.hpp"
#include "utils/dataset.hpp"
#include <stdio.h>
#include <string>

#include <math.h>

//int logits_to_class(float *logits, int num_classes) {
//    float min = -10000;
//    int clas = -1;
//    for(int i = 0; i < num_classes; i++) {
//        if(logits[i] > min) {
//            clas = i;
//        }
//    }
//    return clas;
//}

void print_array(float *arr, int len) {
    for(int i = 0; i < len; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

void print_intarray(int *arr, int len) {
    for(int i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    char fname[] = "../data/data_n100_m5_mu1.5.csv";
    Dataset d(fname, 10);
    layer_type lts[4] = {RELU, Sigmoid, Sigmoid, Softmax};
    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    int epochs = 10;

    Network* my_net = new Network(layout, num_layers, lts);
    my_net->connect();
    for(int epoch = 1; epoch < epochs; epoch++) {
        d.shuffle_sample_order();
        for(int i = 0; i < int(d.n / d.get_batch_size()); i++) {
            d.load_next_batch();
            //printf("minibatch %d:\n", i);
            for(int j = 0; j < d.get_batch_size(); j++) {
                //print_array(d.batch_x[j], 5);
                my_net->set_input(d.batch_x[i]);
                my_net->forward_pass();
                my_net->back_propogate(d.batch_y[i]);
                printf("MSE Loss: %f, \n", MSE(my_net->get_output(), d.batch_y[i], 2));
            }
        }
        printf("\ndone with epoch %d\n", epoch);


        printf("testing network accuracy on one minibatch: \n");
        int correct = 0;
        int pred, actual;
//        int *samp_order = d.get_sample_order()
        for(int i = 0; i < d.get_batch_size(); i++) {
            printf("input: ");
            print_array(d.batch_x[i], 5);
            my_net->set_input(d.batch_x[i]);
            my_net->forward_pass();
            printf("net output: ");
            print_array(my_net->get_output(), 2);
            printf("actual: ");
            print_array(d.batch_y[i], 2);
            printf("\n");
//            pred = logits_to_class(my_net->get_output(), 2);
//            actual = logits_to_class(d.batch_y[i], 2);
//            if(pred == actual)
//                correct += 1;
        }
//        printf("\taccuracy: %f\n", float(correct / d.get_batch_size()));
    }
    delete my_net;
}




