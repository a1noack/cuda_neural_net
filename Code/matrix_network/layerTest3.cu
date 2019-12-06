#include "layer.hpp"

#define BATCH_SZ 1
#define NUM_IN 5
#define NUM_OUT 2
#define MIN_ERR 0.0001
#define MAX_EPOCH 100

int main() {
    Layer* l1 = new Layer(NUM_IN, input, NULL, BATCH_SZ);
    Layer* l2 = new Layer(3, hidden, l1, BATCH_SZ);
    Layer* l3 = new Layer(3, hidden, l2, BATCH_SZ);
    Layer* l4 = new Layer(NUM_OUT, output, l3, BATCH_SZ);

    matrix* targets = new matrix(BATCH_SZ,NUM_OUT);
    targets->move_to_device();
    matrix* temp = new matrix(BATCH_SZ,NUM_OUT);
    tmep->move_to_device();

    l2->move_to_device();
    l3->move_to_device();
    l4->move_to_device();

    float error = 1.0;
    int j = 0;

    while(error > MIN_ERR || j < MAX_EPOCH) {
       j++;
      l1->zero_grad();
      l2->zero_grad();
      l3->zero_grad();
      l4->zero_grad();
    }
}

