#include "layer.hpp"
#include "utils/dataset.hpp"
#include <time.h>

//---------------- Testing file for the serial network------------------
// Architecture:
//      Number of Layers: 3
//      Layout:
//          Input: 5
//          Hidden1: 3
//          Output: 2
//----------------------------------------------------------------------

//----------------GLOBAL PARAMETERS-------------------------------------
#define BATCH_SIZE 10
#define NUM_EPOCHS 100
#define LEARN_RATE 0.1
#define MIN_ERROR 0.0001
#define NUM_INPUTS 1024
#define NUM_OUTPUTS 2

const char* data_file = "../data/data_n1024_m1024_mu2.0.csv";
//----------------------------------------------------------------------

int main() {
    //create layers:
    Layer* l1 = new Layer(NUM_INPUTS, input, NULL);
    Layer* l2 = new Layer(32, hidden, l1);
    Layer* l3 = new Layer(NUM_OUTPUTS, output, l2);

    //Load up the dataset
    Dataset d(data_file, BATCH_SIZE);

    int num_batches = d.n / BATCH_SIZE;

    int j = 0;
    float error = 0.0;

    // Run for the allotted number of epochs
    clock_t start = clock();

    srand(5);
    while(j < NUM_EPOCHS) {
        d.shuffle_sample_order();
        error = 0.0;

        // Do for each batch
        for(int i = 0; i < num_batches; i++) {
            d.load_next_batch();

            l1->zero_grad(); // Zeros out the accumulating gradients
            l2->zero_grad();
            l3->zero_grad();

            // Do for each sample in the batch:
            for(int k = 0; k < BATCH_SIZE; k++) {
                //set inputs from dataset
                l1->outputs->set_memory(d.batch_x[k]);

                //run the forward pass
                l2->forward_pass();
                l3->forward_pass();

                //compute the error
                error += MSE(l3->outputs->host_data, d.batch_y[k], NUM_OUTPUTS);

                //Back propagate
                l3->back_prop(d.batch_y[k]);
                l2->back_prop(NULL);
            }

            //update weights
            l3->update(LEARN_RATE, BATCH_SIZE);
            l2->update(LEARN_RATE, BATCH_SIZE);
        }

        //compute error
        error /= (float)num_batches * BATCH_SIZE;

        printf("Epoch #%d, Error = %f\n", j, error);
        fflush(stdout);
        j++;
    }
    double elapsed_time = double(clock() - start) / CLOCKS_PER_SEC;

    printf("TRAINING NETWORK: [1025,32,2], Batch Size: %d, Epochs: %d, Error: %f, Training Time: %f Seconds\n", BATCH_SIZE, NUM_EPOCHS, error, elapsed_time);
}


