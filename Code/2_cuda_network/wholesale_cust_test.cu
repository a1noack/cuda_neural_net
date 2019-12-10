#include "layer.hpp"
#include "utils/dataset.hpp"
#include <time.h>

#define BATCH_SZ 10
#define NUM_IN 7
#define NUM_OUT 2
#define MIN_ERR 0.001
#define MAX_EPOCH 75
#define LEARN_RATE 2

int main() {
    clock_t t;
    srand(5);

    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(8, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(8, hidden, l2, BATCH_SZ);
    Layer* l4 = new Layer(8, hidden, l3, BATCH_SZ);
    Layer* l5 = new Layer(NUM_OUT, output, l4, BATCH_SZ);

    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    temp->move_to_device();

    l2->move_to_device();
    l3->move_to_device();
    l4->move_to_device();
    l5->move_to_device();

    float error = 1.0;
    int j = 0;

    char* file_name = "../data/wholesale_cust_mean.csv";

    Dataset d(file_name, BATCH_SZ);
    int total_samples = d.n;
    int num_batches = d.n / BATCH_SZ;
    float learn_rate = LEARN_RATE;
    
    t = clock();
    while(j < MAX_EPOCH) {
       if(j % 30 == 0)
           learn_rate /= 1.5;
       d.shuffle_sample_order();
       error = 0;

       for(int i = 0; i < num_batches; i++) {
           d.load_next_batch();

           l1->zero_grad();
           l2->zero_grad();
           l3->zero_grad();
           l4->zero_grad();
           l5->zero_grad();

           l1->outputs->set_data_loader(d.batch_x);
           targets->set_data_loader(d.batch_y);

           l2->forward_pass();
           l3->forward_pass();
           l4->forward_pass();
           l5->forward_pass();

           if(i % 100 == 0 and j > MAX_EPOCH - 2){
                printf("targets: \n"); targets->print();
                printf("outputs: \n"); l5->outputs->print(); printf("\n\n");
           }

           error += MSE_mat_wrapper(targets, l5->outputs, temp);

           l5->back_prop(targets, BATCH_SZ);
           l4->back_prop(NULL, BATCH_SZ);
           l3->back_prop(NULL, BATCH_SZ);
           l2->back_prop(NULL, BATCH_SZ);
        
           l5->update(learn_rate, BATCH_SZ);
           l4->update(learn_rate, BATCH_SZ);
           l3->update(learn_rate, BATCH_SZ);
           l2->update(learn_rate, BATCH_SZ);
       }
       error /= (float)num_batches;
       printf("Epoch #%d, Error = %f\n", j, error);
       fflush(stdout);
       j++;
    }
    double elapsed_seconds = double(clock() - t) / CLOCKS_PER_SEC;
    printf("time = %f seconds\n", elapsed_seconds);

    printf("TRAINING SUSPENDED AT: EPOCH #%d, ERROR: %f\n", j, error);
    fflush(stdout);
}

