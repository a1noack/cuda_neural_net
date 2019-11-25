#include "nn.hpp"
#include "utils/dataset.hpp"
#include <stdio.h>
#define BATCH_SIZE 5
#define EPOCHS 5
#define MIN_ERR 0.0001
#define LEARNING_RATE 0.1
float average_err(float* errors, int num) {
    if(num == 1) {
        return errors[0];
    }

    float sum_errs = 0.0;
    for(int i = 0; i < num; i++) {
        sum_errs += errors[i];
    }
    return sum_errs / num;
}

void print_float_arr(float* a, int n) {
    for(int i = 0; i < n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
}

int main(int argv, char** argc) {

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";

    Dataset d(file_name, BATCH_SIZE);
    int total_samples = d.n;
    int num_batches = d.n / BATCH_SIZE;
    int layout[4] = {5,7,5,2};
    int num_layers = 4;
    layer_type lts[4] = {Sigmoid, Sigmoid, Sigmoid, Sigmoid};

    Network* my_net = new Network(layout, num_layers, lts);

    my_net->connect();

    float cur_error = 0.0;
    int cur_epoch, cur_batch = 0;
    for(int i = 0; i < EPOCHS; i++) {
        printf("Epoch #%d, Current error: %f\n", i+1, cur_error);
        d.shuffle_sample_order();
        cur_epoch = i;
        for(int j = 0; j < num_batches; j++) {
            printf("Batch#%d, Current error: %f\n", j+1, cur_error);
            d.load_next_batch();
            my_net->zero_grad();
            float errors[BATCH_SIZE] = {};
            for(int k = 0; k < BATCH_SIZE; k++) {
                my_net->set_input(d.batch_x[k]);
                my_net->forward_pass();
                //printf("Outputs:\n");
                //print_float_arr(my_net->get_output(), 2);
                //printf("Targets:\n");
                //print_float_arr(d.batch_y[k], 2);
                //printf("Before mse:\n");
                errors[k] = MSE(my_net->get_output(), d.batch_y[k], 2);
                my_net->back_propogate(d.batch_y[k]); // <-------- Do we back prop with the last results from the batch? or use MSE somehow?
            }
            cur_batch = j;
            //printf("cur error calc\n");
            cur_error = average_err(errors, BATCH_SIZE);
            if( cur_error <= MIN_ERR) goto done_training;
            //printf("Backprop\n");
            //print_float_arr(d.batch_y[0], 2);
            //printf("updates\n");
            my_net->update_weights(LEARNING_RATE, BATCH_SIZE);
        }
    }

    done_training:
    printf("Training at Epoch#%d Batch#%d, has finished with error: %f\n", cur_epoch, cur_batch, cur_error);
    delete my_net;
}






