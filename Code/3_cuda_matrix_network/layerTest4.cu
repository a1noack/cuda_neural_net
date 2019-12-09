#include "layer.hpp"
#include "utils/dataset.hpp"

#define BATCH_SZ 1
#define NUM_IN 5
#define NUM_OUT 2
#define MIN_ERR 0.001
#define MAX_EPOCH 1
#define LEARN_RATE 0.01

int main() {

    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(2, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(NUM_OUT, output, l2, BATCH_SZ);

    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    temp->move_to_device();

    l2->move_to_device();
    l3->move_to_device();

    float error = 1.0;
    int j = 0;

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";

    Dataset d(file_name, BATCH_SZ);
    int total_samples = d.n;
    int num_batches = d.n / BATCH_SZ;


    while(j < MAX_EPOCH) {
       d.shuffle_sample_order();

       for(int i = 0; i < num_batches; i++) {
           //batch
           d.load_next_batch();

           l1->zero_grad();
           l2->zero_grad();
           l3->zero_grad();

           l1->outputs->set_data_loader(d.batch_x);
           targets->set_data_loader(d.batch_y);

           l2->forward_pass();
           l3->forward_pass();

/*           if( j % 10 == 0) {
               printf("Printing layer 2:\n");
                l2->print_layer();
               printf("Printing layer 3:\n");
                l3->print_layer();
               printf("Printing targets:\n");
                targets->print();
           }
*/

           error = MSE_mat_wrapper(targets, l3->outputs, temp);
           if (error < MIN_ERR) { break; }

            printf("Batch #%d, Error = %f\n", i, error);
           l3->back_prop(targets, BATCH_SZ);
           l2->back_prop(NULL, BATCH_SZ);

           l3->update(LEARN_RATE, BATCH_SZ);
           l2->update(LEARN_RATE, BATCH_SZ);
       }
       printf("Epoch #%d, Error = %f\n", j, error);
    fflush(stdout);
       j++;
    }

    printf("TRAINING SUSPENDED AT: EPOCH #%d, ERROR: %f\n", j, error);
    fflush(stdout);
}

