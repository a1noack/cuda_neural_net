#include "layer.hpp"
#include "utils/dataset.hpp"
#include <time.h>

#define BATCH_SZ 1024
#define NUM_IN 1024
#define NUM_OUT 2
#define MIN_ERR 0.0001
#define MAX_EPOCH 100
#define LEARN_RATE 0.1

int main() {
    clock_t t;
    srand(5);

    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(32, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(NUM_OUT, output, l2, BATCH_SZ);

    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    temp->move_to_device();

    l2->move_to_device();
    l3->move_to_device();

    float error = 0;
    int j = 0;

    char* file_name = "../data/data_n2048_m1024_mu2.0.csv";

    Dataset d(file_name, BATCH_SZ);
    int total_samples = d.n;
    int num_batches = d.n / BATCH_SZ;

    t = clock();
    while(j < MAX_EPOCH) {
       error = 0;
       d.shuffle_sample_order();

       for(int i = 0; i < num_batches; i++) {
           d.load_next_batch();

           l1->zero_grad();
           l2->zero_grad();
           l3->zero_grad();

           l1->outputs->set_data_loader(d.batch_x);
           targets->set_data_loader(d.batch_y);

           l2->forward_pass();
           l3->forward_pass();

           error += MSE_mat_wrapper(l3->outputs, targets, temp);
           if (error < MIN_ERR) { break; }

           l3->back_prop(targets, BATCH_SZ);
           l2->back_prop(NULL, BATCH_SZ);

           l3->update(LEARN_RATE, BATCH_SZ);
           l2->update(LEARN_RATE, BATCH_SZ);
       }
       printf("Epoch #%d, Error = %f\n", j, error);
       fflush(stdout);
       j++;
    }
    double elapsed_seconds = double(clock() - t) / CLOCKS_PER_SEC;
    printf("time = %f seconds\n", elapsed_seconds);

    printf("TRAINING SUSPENDED AT: EPOCH #%d, ERROR: %f\n", j, error);
    fflush(stdout);
}

