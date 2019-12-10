#include "layer.hpp"
#include "utils/dataset.hpp"
#include <time.h>

//---------------- Testing file for the serial network------------------
// Architecture:
//      Number of Layers: 9
//      Layout:
//          Input: 5
//          Hidden1: 7
//          Hidden2: 10
//          Hidden3: 15
//          Hidden4: 20
//          Hidden5: 15
//          Hidden6: 10
//          Hidden7: 7
//          Output: 2
//----------------------------------------------------------------------

//----------------GLOBAL PARAMETERS-------------------------------------
#define BATCH_SIZE 10
#define NUM_EPOCHS 100
#define LEARN_RATE 0.1
#define MIN_ERROR 0.0001
#define NUM_INPUTS 5
#define NUM_OUTPUTS 2

const char* data_file = "../data/data_n1000_m5_mu1.5.csv";
//----------------------------------------------------------------------

int main() {
    //create layers:
    Layer* l1 = new Layer(NUM_INPUTS, input, NULL);
    Layer* l2 = new Layer(7, hidden, l1);
    Layer* l3 = new Layer(10, hidden, l2);
    Layer* l4 = new Layer(15, hidden, l3);
    Layer* l5 = new Layer(20, hidden, l4);
    Layer* l6 = new Layer(15, hidden, l5);
    Layer* l7 = new Layer(10, hidden, l6);
    Layer* l8 = new Layer(7, hidden, l7);
    Layer* l9 = new Layer(NUM_OUTPUTS, output, l8);

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
            l6->zero_grad();
            l7->zero_grad();
            l8->zero_grad();
            l9->zero_grad();

            // Do for each sample in the batch:
            for(int k = 0; k < BATCH_SIZE; k++) {
                //set inputs from dataset
                l1->outputs->set_memory(d.batch_x[k]);

                //run the forward pass
                l2->forward_pass();
                l3->forward_pass();
                l4->forward_pass();
                l5->forward_pass();
                l6->forward_pass();
                l7->forward_pass();
                l8->forward_pass();
                l9->forward_pass();

                //compute the error
                error += MSE(l9->outputs->host_data, d.batch_y[k], NUM_OUTPUTS);

                //Back propagate
                l9->back_prop(d.batch_y[k]);
                l8->back_prop(NULL);
                l7->back_prop(NULL);
                l6->back_prop(NULL);
                l5->back_prop(NULL);
                l4->back_prop(NULL);
                l3->back_prop(NULL);
                l2->back_prop(NULL);
            }

            //update weights
            l9->update(LEARN_RATE, BATCH_SIZE);
            l8->update(LEARN_RATE, BATCH_SIZE);
            l7->update(LEARN_RATE, BATCH_SIZE);
            l6->update(LEARN_RATE, BATCH_SIZE);
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

    printf("TRAINING NETWORK: [5,7,10,15,20,10,15,7,2], Batch Size: %d, Epochs: %d, Error Threshold: %f, Error: %f, Training Time: %f Seconds\n", BATCH_SIZE, NUM_EPOCHS, MIN_ERROR, error, elapsed_time);
}


