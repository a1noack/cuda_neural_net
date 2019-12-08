#include "layer.hpp"
#include "utils/dataset.hpp"

#define BATCH_SZ 10
#define NUM_IN 5
#define NUM_OUT 2
#define MIN_ERR 0.001
#define MAX_EPOCH 1
#define LEARN_RATE 0.01

int main() {

    //printf("SETTING LAYERS\n");
    //fflush(stdout);
    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(7, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(NUM_OUT, output, l2, BATCH_SZ);


    //printf("CREATING TARGETS\n");
    //fflush(stdout);
    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    temp->move_to_device();

    //printf("MOVE TO DEVICE\n");
    //fflush(stdout);
    l2->move_to_device();
    l3->move_to_device();

    float error = 1.0;
    int j = 0;

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";


    //printf("CREATE DATASET\n");
    //fflush(stdout);
    Dataset d(file_name, BATCH_SZ);
    int total_samples = d.n;
    int num_batches = d.n / BATCH_SZ;


    while(j < MAX_EPOCH) {
        //printf("SHUFFLING....\n");
        //fflush(stdout);
       d.shuffle_sample_order();

       for(int i = 0; i < num_batches; i++) {
           //batch
          // printf("LOADING BATCH\n");
           //fflush(stdout);
           d.load_next_batch();

           //printf("ZERO GRADS\n");
    //fflush(stdout);
           l1->zero_grad();
           l2->zero_grad();
           l3->zero_grad();

           //printf("SETTING BATCH DATA\n");
    //fflush(stdout);
           l1->outputs->set_data_loader(d.batch_x);
           targets->set_data_loader(d.batch_y);

      //     printf("FORWARD\n");
    //fflush(stdout);
           l2->forward_pass();
           l3->forward_pass();

           if( j % 10 == 0) {
               printf("Printing layer 2:\n");
                l2->print_layer();
               printf("Printing layer 3:\n");
                l3->print_layer();
               printf("Printing targets:\n");
                targets->print();
           }


      //     printf("ERROR\n");
    ///fflush(stdout);
           error = MSE_mat_wrapper(l3->outputs, targets, temp);
           if (error < MIN_ERR) { break; }

       //    printf("BACK_PROP\n");
    //fflush(stdout);
           l3->back_prop(targets, BATCH_SZ);
           l2->back_prop(NULL, BATCH_SZ);

                 //     printf("UPDATES\n");
    //fflush(stdout);
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

