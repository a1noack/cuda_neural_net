#include "softmax_layer.hpp"

/* Layer constructor. Initializes many data members. What it cannot it will set to NULL, hopefully avoid shitty behavior */
Softmax_Layer::Softmax_Layer(int numNodes) {
    this->num_nodes = numNodes;
    prev_layer = NULL;
    next_layer = NULL;

    weights = NULL;
    outputs = new float[numNodes];

    bias = new float[numNodes];
    del_bias = new float[numNodes];

    for(int i = 0; i < numNodes; i++) {
        bias[i] = get_random_f();
        outputs[i] = 0.0;
        del_bias[i] = 0.0;
    }

    printf("Softmax layer Created %d nodes\n", this->num_nodes);
}

Softmax_Layer::~Softmax_Layer() {
    if(weights != NULL) {
        delete [] weights;
        delete [] del_weights;
    }

    delete [] outputs;
    delete [] bias;
    delete [] del_bias;
    delete [] dp;
}

void Softmax_Layer::forward_pass(Layer* prev) {
    float* dpros = new float[num_nodes];

    for(int i = 0; i < num_nodes; i++) {
        dpros[i]= dot_prod(weights[i], prev->get_outputs(), num_nodes);
    }

    float e_sum = 0.0;
    for(int i = 0; i < num_nodes; i++) {
        e_sum += expf(dpros[i]);
    }

    for(int i = 0; i < num_nodes; i++) {
        outputs[i] = expf(dpros[i]) / e_sum;
    }
    dp = dpros;
    esum = e_sum;
}

/* The good shit. Back propogate the error on just the input layer. */
void Softmax_Layer::back_prop_input(float* targets) {
    int num_weights = prev_layer->get_num_nodes();

    for(int i = 0; i < num_nodes; i++) {
        float o = outputs[i];

        float e_sum = esum - expf(dp[i]);
        del_bias[i] += ( expf(dp[i]) * e_sum ) / powf(esum, 2);

        float* dw = new float[num_weights];
        float* po = prev_layer->get_outputs();

        for(int j = 0; j < num_weights; j++) {

            dw[j] += del_bias[i] * po[j];
        }

        del_weights[i] = dw;
    }

}

/* the really good shit. Back propgates the error for the hidden layers */
void Softmax_Layer::back_prop() {
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
        float e_sum = esum - expf(dp[i]);
        float dr = ( expf(dp[i]) * e_sum ) / powf(esum, 2);

        del_bias[i] += db * dr;
        //printf("fdb:::%f\n", del_bias[i]);
        float* dw = new float[num_nodes];

        float* po = prev_layer->get_outputs();

        for(int k = 0; k < num_nodes; k++) {
            dw[k] += del_bias[i] * po[k];
          //  printf("dw = %f, db = %f, po = %f\n", dw[k], del_bias[i], po[k]);
        }

        del_weights[i] = dw;
    }
}
