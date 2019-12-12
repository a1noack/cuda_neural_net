#include "layer.hpp"
#include "utils/dataset.hpp"
#include <time.h>

//---------------- Testing file for the serial network------------------
// Architecture:
//      Number of Layers: 5
//      Layout:
//          Input: 5
//          Hidden1: 8
//          Hidden2: 8
//          Hidden3: 8
//          Output: 2
//----------------------------------------------------------------------

//----------------GLOBAL PARAMETERS-------------------------------------
#define BATCH_SIZE 10
#define NUM_EPOCHS 100
#define LEARN_RATE 0.01
#define MIN_ERROR 0.001
#define NUM_INPUTS 7
#define NUM_OUTPUTS 2

const char* data_file = "../data/wholesale_cust_mean.csv";
//----------------------------------------------------------------------

int main() {
    //create layers:
    Layer* l1 = new Layer(NUM_INPUTS, input, NULL);
    Layer* l2 = new Layer(8, hidden, l1);
    Layer* l3 = new Layer(8, hidden, l2);
    Layer* l4 = new Layer(8, hidden, l3);
    Layer* l5 = new Layer(NUM_OUTPUTS, output, l4);

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
            l4->zero_grad();
            l5->zero_grad();

            // Do for each sample in the batch:
            for(int k = 0; k < BATCH_SIZE; k++) {
                //set inputs from dataset
                l1->outputs->set_memory(d.batch_x[k]);

                //run the forward pass
                l2->forward_pass();
                l3->forward_pass();
                l4->forward_pass();
                l5->forward_pass();

                //compute the error
                error += MSE(l5->outputs->host_data, d.batch_y[k], NUM_OUTPUTS);

                //Back propagate
                l5->back_prop(d.batch_y[k]);
                l4->back_prop(NULL);
                l3->back_prop(NULL);
                l2->back_prop(NULL);
            }

            //update weights
            l5->update(LEARN_RATE, BATCH_SIZE);
            l4->update(LEARN_RATE, BATCH_SIZE);
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

    printf("TRAINING NETWORK: [5,8,8,8,2], Batch Size: %d, Epochs: %d, Error Threshold: %f, Error: %f, Training Time: %f Seconds\n", BATCH_SIZE, NUM_EPOCHS, MIN_ERROR, error, elapsed_time);
}


