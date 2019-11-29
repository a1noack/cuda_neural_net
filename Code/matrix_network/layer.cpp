#include "layer.hpp"

Layer::Layer(int node_count, layer_pos lpp, Layer* previous_layer) {
    lp = lpp;
    num_nodes = node_count;

    prev = NULL;
    next = NULL;
    in_weights = NULL;
    in_del_weights = NULL;
    out_weights = NULL;
    out_del_weights = NULL;

    outputs = new matrix(1, num_nodes);
    inputs = new matrix(1, num_nodes);

    if(previous_layer != NULL) {
        prev = previous_layer;
        prev->set_next_layer(this);
        in_weights = prev->out_weights;
        in_del_weights = prev->out_del_weights;
    }

    bias = new matrix(1, node_count);
    bias->set_mem_random();

    del_bias = new matrix(1, node_count);
    del_bias->set_mem_zero();

}

void Layer::set_next_layer(Layer* next_layer) {

    next = next_layer;
    out_weights = new matrix(next->num_nodes, num_nodes);
    next->in_weights = out_weights;

    out_del_weights = new matrix(next->num_nodes, num_nodes);
    next->in_del_weights = out_del_weights;
}

void Layer::zero_grad() {
    if(lp != output) {
        del_bias->set_mem_zero();
        out_del_weights->set_mem_zero();
    }
}

void Layer::forward_pass() {
    if(lp != output) {

        for(int i = 0; i < next->num_nodes; i++) {
            float dp = dot_prod(out_weights->get_row(i), inputs->get_row(0), next->num_nodes);
            *outputs->get_row(0)[i] = ( 1 / ( 1 + expf( -1 * (dp + *bias->get_row(0)[i] ) ) ) );
        }
    }
}

void Layer::back_prop(float* targets) {
    for(int i = 0; i < num_nodes; i++) {

        float o = *outputs->get_row(0)[i];
        float dbloc = (o * (1-o));

        if(targets != NULL) {
            dbloc *= o - targets[i];
        } else {
            float** w = out_weights->get_col(i);
            float** ndb = next->del_bias->get_row(0);

            float db = 0.0;
            for(int j = 0; j < out_weights->num_cols; j++) {
                db = *ndb[j] + *w[j];
            }

            dbloc *= db;
        }

        *del_bias->get_row(0)[i] += dbloc;

        float** dw = in_del_weights->get_row(i);

        float** ins = inputs->get_row(0);

        for(int j = 0; j < in_del_weights->num_cols; j++) {
            *dw[j] += dbloc * *ins[j];
        }
    }
}


void Layer::update(float learn_rate, int batch_size) {
    float** b = bias->get_row(0);
    float** db = del_bias->get_row(0);

    for(int i = 0; i < num_nodes; i++) {
        float** w = in_weights->get_row(i);
        float** dw = in_del_weights->get_row(i);

        for(int j = 0; j < prev->num_nodes; j++) {
            *w[j] = *w[j] - (learn_rate * (*dw[j] / batch_size) );
        }

        *b[i] = *b[i] - (learn_rate * (*db[i] / batch_size) );
    }
}

float dot_prod(float* x, float* y, int num) {
    float dp = 0.0;

    for(int i = 0; i < num; i++) {
        dp += x[i] * y[i];
    }
    return dp;
}

float dot_prod(float** x, float** y, int num) {
    float dp = 0.0;

    for(int i = 0; i < num; i++) {
        dp += *x[i] * *y[i];
    }
    return dp;
}

float MSE(float* v1, float* v2, int num) {
    float s = 0.0;

    for(int i = 0; i < num; i++) {
        s += pow( (double) v1[i] - v2[i], 2);
    }

    return ( (float) 1 / (float) num ) * s;
}

float MSE(float** v1, float** v2, int num) {
    float s = 0.0;

    for(int i = 0; i < num; i++) {
        s += pow( (double) *v1[i] - *v2[i], 2);
    }

    return ( (float) 1 / (float) num ) * s;
}


