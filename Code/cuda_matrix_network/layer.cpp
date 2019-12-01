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

    n_idx = new matrix(1, node_count);
    float* idxs = new float[node_count];

    for(int i = 0; i < node_count; i++) {
        idxs[i] = i+1;
    }

    n_idx->set_memory(idxs);

}

void Layer::set_next_layer(Layer* next_layer) {

    next = next_layer;
    out_weights = new matrix(num_nodes, next->num_nodes);
    out_weights->set_mem_random();

    next->in_weights = out_weights;
    next->inputs = outputs;

    out_del_weights = new matrix(num_nodes, next->num_nodes);
    out_del_weights->set_mem_zero();
    next->in_del_weights = out_del_weights;

    //____________________TESTING STUFF
    w_idx = new matrix(num_nodes, next->num_nodes);
    float* idxs = new float[num_nodes * next->num_nodes];

    for(int i = 0; i < num_nodes * next->num_nodes; i++) {
        idxs[i] = i + 1;
    }

    w_idx->set_memory(idxs);

}

void Layer::zero_grad() {
    if(lp != output) {
        del_bias->set_mem_zero();
        out_del_weights->set_mem_zero();
    }
}

void print_FF(float** f, int n) {
    for(int i = 0; i < n; i++) {
        printf("%f ", *f[i]);
    }
    printf("\n");
}

void Layer::forward_pass() {
    if(lp != input) {

        for(int i = 0; i < num_nodes; i++) {

            assert(in_weights->num_rows == inputs->num_cols);
            //<---------- COLUMN CALL
            float dp = dot_prod(in_weights->get_col(i), inputs->get_row(0), in_weights->num_rows);
            *outputs->get_row(0)[i] = ( 1 / ( 1 + expf( -1 * (dp + *bias->get_row(0)[i] ) ) ) );
        }
    }
}

void Layer::back_prop(float* targets) {
    for(int i = 0; i < num_nodes; i++) {
        assert(num_nodes == outputs->num_cols);
        float o = *outputs->get_row(0)[i];
        float dbloc = (o * (1-o));

        if(targets != NULL) {
            dbloc *= (o - targets[i]);
        } else {
            float** w = out_weights->get_row(i);
            float** ndb = next->del_bias->get_row(0);
            float db = 0.0;
            assert(out_weights->num_cols == next->del_bias->num_cols);
            for(int j = 0; j < out_weights->num_cols; j++) {
                db += *ndb[j] * *w[j];
            }
            dbloc *= db;
        }

        *del_bias->get_row(0)[i] += dbloc;

        float** dw = in_del_weights->get_col(i); //<------------------ COLUMN CALL

        float** ins = inputs->get_row(0);

        assert(in_del_weights->num_rows == inputs->num_cols);
        for(int j = 0; j < in_del_weights->num_rows; j++) {
            *dw[j] += dbloc * *ins[j];
        }
    }
}


void Layer::update(float learn_rate, int batch_size) {
    float** b = bias->get_row(0);
    float** db = del_bias->get_row(0);

    assert(num_nodes == in_weights->num_cols);
    for(int i = 0; i < num_nodes; i++) {
        float** w = in_weights->get_col(i); //<--------------- COLUMN CALLS
        float** dw = in_del_weights->get_col(i);

        assert(prev->num_nodes == in_weights->num_rows);
        for(int j = 0; j < prev->num_nodes; j++) {
            *w[j] = *w[j] - (learn_rate * (*dw[j] / batch_size) );
        }

        *b[i] = *b[i] - (learn_rate * (*db[i] / (float)batch_size) );
    }
}

void Layer::print_layer() {
    printf("printing layer\n");
    if(inputs != NULL) {
        printf("Inputs:\n");
        inputs->print();
        printf("\n");
    }
    if(in_weights != NULL) {
        printf("in_weights:\n");
        in_weights->print();
        printf("\n");
    }
    if(in_del_weights != NULL) {
        printf("in_del_w:\n");
        in_del_weights->print();
        printf("\n");
    }
    if(outputs != NULL) {
        printf("outputs:\n");
        outputs->print();
        printf("\n");
    }
    if(out_weights != NULL) {
        printf("oput_weights:\n");
        out_weights->print();
        printf("\n");
    }
    if(out_del_weights != NULL) {
        printf("out_del_w:\n");
        out_del_weights->print();
        printf("\n");
    }
    if(bias != NULL) {
        printf("Bias:\n");
        bias->print();
        printf("\n");
    }
    if(del_bias != NULL) {
        printf("del_bias:\n");
        del_bias->print();
        printf("\n");
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
    //printf("in dp\n");
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

float MSE(float** v1, float* v2, int num) {
    float s = 0.0;

    for(int i = 0; i < num; i++) {
        s += pow( (double) *v1[i] - v2[i], 2);
    }

    return ( (float) 1 / (float) num ) * s;
}


