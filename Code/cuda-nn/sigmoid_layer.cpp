#include "sigmoid_layer.hpp"

/* Layer constructor. Initializes many data members. What it cannot it will set to NULL, hopefully avoid shitty behavior */
Sigmoid_Layer::Sigmoid_Layer(int numNodes) {
    this->num_nodes = numNodes;
    prev_layer = NULL;
    next_layer = NULL;

    weights = NULL;

    /*
    outputs = new float[numNodes];
    bias = new float[numNodes];
    del_bias = new float[numNodes];
    */

    outputs = new matrix(1, numNodes);
    bias = new matrix(1, numNodes);
    del_bias = new matrix(1, numNodes);

    /*
    for(int i = 0; i < numNodes; i++) {
        bias[i] = get_random_f();
        outputs[i] = 0.0;
        del_bias[i] = 0.0;
    }
    */

    outputs->set_mem_zero(outputs->dim_x, outputs->dim_y);
    bias->set_mem_random(bias->dim_x, outputs->dim_y);
    del_bias->set_mem_zero(del_bias->dim_x, del_bias->dim_y);

    printf("Sigmoid layer Created %d nodes\n", this->num_nodes);
}

Sigmoid_Layer::~Sigmoid_Layer() {
    if(weights != NULL) {
        delete weights;
        delete del_weights;
    }

    delete outputs;
    delete bias;
    delete del_bias;
}

void Sigmoid_Layer::forward_pass(Layer* prev) {
    /*
    for(int i = 0; i < num_nodes; i++) {
        float dp = dot_prod(weights[i], prev->get_outputs(), num_nodes);
        outputs[i] = (1 / (1+ expf( -1* (dp + bias[i]) ) ) );
    }
    */
    float* opts = outputs->host_data;
    for(int i = 0; i < num_nodes; i++) {
        float dp = dot_prod(weights->host_data[i], prev->get_outputs()->host_data, num_nodes);
        opts[i] = (1/ (1+ expf( -1 * (dp + bias->host_data[i]))));
    }

}

/* The good shit. Back propogate the error on just the input layer. */
void Sigmoid_Layer::back_prop_input(float* targets) {
    int num_weights = prev_layer->get_num_nodes();

    for(int i = 0; i < num_nodes; i++) {
        float o = outputs[i];

        float dbloc = (o - targets[i]) * (o * (1.0 - o));

        del_bias[i] += dbloc;
        //printf("delb = %f\n", del_bias[i]);

        float* dw = del_weights[i];//new float[num_weights];
        float* po = prev_layer->get_outputs();

        for(int j = 0; j < num_weights; j++) {
            dw[j] += dbloc * po[j];
            //printf("dw = %f, (bd = %f, pw = %f\n", dw[j], del_bias[i], po[j]);
        }

        //del_weights[i] = dw; //<------- MEMORY LEAK HERE
    }
    //printf("\n");

}

/* the really good shit. Back propgates the error for the hidden layers */
void Sigmoid_Layer::back_prop() {

    int num_weights = prev_layer->get_num_nodes();
    float* ndb = next_layer->get_del_bias();

    for(int i = 0; i < num_nodes; i++) {
        float** w = next_layer->get_weights();

        int next_num_weights = next_layer->get_num_nodes();
        float db = 0.0;

        for(int j = 0; j < next_layer->get_num_nodes(); j++) {

            db += ndb[j] * w[j][i];
        }

        //printf("delb:::%f\n", db);
        float o = outputs[i];
        //printf("out: %f\n", o);
        float dbloc = db * (o * (1.0-o));

        del_bias[i] += dbloc;
        //printf("fdb:::%f\n", del_bias[i]);
        float* dw = del_weights[i];//new float[num_nodes];

        float* po = prev_layer->get_outputs();

        //printf("delweights: \n");
        for(int k = 0; k < num_nodes; k++) {
            dw[k] += dbloc * po[k];
          //  printf("%f ", dw[k]);
          //  printf("dw = %f, db = %f, po = %f\n", dw[k], del_bias[i], po[k]);
        }
        //printf("\n");
        //del_weights[i] = dw; //<------ MEMORY LEAK
    }
}
