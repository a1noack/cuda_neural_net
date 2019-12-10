#include "layer.hpp"
#include "utils/dataset.hpp"
#include <cuda_runtime.h>
#include <omp.h>

#define BATCH_SZ 10
#define NUM_IN 5
#define NUM_OUT 2
#define MIN_ERR 0.001
#define MAX_EPOCH 10
#define LEARN_RATE 0.1
#define NUM_GPU 2

typedef struct {

    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(3, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(3, hidden, l2, BATCH_SZ);
    Layer* l4 = new Layer(NUM_OUT, output, l3, BATCH_SZ);


    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
} gpu_net;

int main() {

    gpu_net dev_0;
    gpu_net dev_1;


    /*
    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(3, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(3, hidden, l2, BATCH_SZ);
    Layer* l4 = new Layer(NUM_OUT, output, l3, BATCH_SZ);


    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    temp->move_to_device();
*/
    cudaSetDevice(0);
    dev_0.l2->move_to_device();
    dev_0.l3->move_to_device();
    dev_0.l4->move_to_device();
    dev_0.targets->move_to_device();
    dev_0.temp->move_to_device();

    cudaSetDevice(1);
    dev_1.l2->move_to_device();
    dev_1.l3->move_to_device();
    dev_1.l4->move_to_device();
    dev_1.targets->move_to_device();
    dev_1.temp->move_to_device();

    gpu_net gpus[2] = {dev_0, dev_1};

    float error = 1.0;
    float error2 = 1.0;

    int j = 0;

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";
    Dataset d(file_name, BATCH_SZ);

    int total_samples = d.n;
    int num_batches = d.n / BATCH_SZ;

    printf("WTF!\n");

    while(j < MAX_EPOCH) {
        printf("Epoch#: %d\n", j);
       d.shuffle_sample_order();

       for(int i = 0; i < num_batches; i++) {
           printf("ITR: %d\n", i);
           error = 0.0;
       int k = 0;
       d.load_next_batch();
           for(k = 0; k < NUM_GPU; k++) {
               cudaSetDevice(k);
               //d.load_next_batch();
               printf("BATCH #%d\n", k);
               gpus[k].l1->zero_grad();
               gpus[k].l2->zero_grad();
               gpus[k].l3->zero_grad();
               gpus[k].l4->zero_grad();

               gpus[k].l1->outputs->set_data_loader(d.batch_x);
               gpus[k].targets->set_data_loader(d.batch_y);

               gpus[k].l2->forward_pass();
               gpus[k].l3->forward_pass();
               gpus[k].l4->forward_pass();

               error = error + MSE_mat_wrapper(gpus[k].l4->outputs, gpus[k].targets, gpus[k].temp);
               //if (error < MIN_ERR) { break; }

               gpus[k].l4->back_prop(gpus[k].targets, BATCH_SZ);
               gpus[k].l3->back_prop(NULL, BATCH_SZ);
               gpus[k].l2->back_prop(NULL, BATCH_SZ);

               gpus[k].l4->update(LEARN_RATE, BATCH_SZ);
               gpus[k].l3->update(LEARN_RATE, BATCH_SZ);
               gpus[k].l2->update(LEARN_RATE, BATCH_SZ);
            }
       }

       printf("Epoch #%d, Error = %f\n", j, error/BATCH_SZ);
    fflush(stdout);
       j++;
    }

    printf("TRAINING SUSPENDED AT: EPOCH #%d, ERROR: %f\n", j, error/BATCH_SZ);
    fflush(stdout);
}

