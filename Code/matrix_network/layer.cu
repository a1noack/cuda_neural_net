#include "layer.hpp"
#include "cuda_kernels.cu"

Layer::Layer(int node_count, layer_pos lpp, Layer* previous_layer, int batch_sz) {
    lp = lpp;
    num_nodes = node_count;

    prev = NULL;
    next = NULL;
    in_weights = NULL;
    in_del_weights = NULL;
    out_weights = NULL;
    out_del_weights = NULL;
    out_weightsT = NULL;


    outputs = new matrix(batch_sz, num_nodes);
    raw_outputs = new matrix(batch_sz, num_nodes);
    //inputs = new matrix(batch_sz, num_nodes);

    inputs = NULL;
    inputsT = NULL;

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
    next->inputsT = new matrix(outputs->num_cols, outputs->num_rows);

    out_del_weights = new matrix(num_nodes, next->num_nodes);
    out_del_weights->set_mem_zero();
    next->in_del_weights = out_del_weights;

    out_weightsT = new matrix(next->num_nodes, num_nodes);

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

void Layer::move_to_device() {
    if(inputs != NULL)
        inputs->move_to_device();
    if(outputs != NULL) {
        outputs->move_to_device();
        raw_outputs->move_to_device();
    }

    if(in_weights != NULL) {
        in_weights->move_to_device();
        in_del_weights->move_to_device();
    }
    if(out_weights != NULL) {
        out_weights->move_to_device();
        out_del_weights->move_to_device();
    }
    if(bias != NULL) {
        bias->move_to_device();
        del_bias->move_to_device();
    }
}

void Layer::move_to_host() {
    if(inputs != NULL)
        inputs->move_to_host();
    if(outputs != NULL) {
        outputs->move_to_host();
        raw_outputs->move_to_host();
    }
    if(in_weights != NULL) {
        in_weights->move_to_host();
        in_del_weights->move_to_host();
    }
    if(out_weights != NULL) {
        out_weights->move_to_host();
        out_del_weights->move_to_host();
    }
    if(bias != NULL) {
        bias->move_to_host();
        del_bias->move_to_host();
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
        mat_mul(inputs, in_weights, raw_outputs);
        add_bias(raw_outputs, bias);
        //raw_outputs->mat_copy_from(outputs);
        activate(raw_outputs, outputs, 0);
    }
    /*    if(lp != input) {

        for(int i = 0; i < num_nodes; i++) {

            assert(in_weights->num_rows == inputs->num_cols);
            //<---------- COLUMN CALL
            float dp = dot_prod(in_weights->get_col(i), inputs->get_row(0), in_weights->num_rows);
            *outputs->get_row(0)[i] = ( 1 / ( 1 + expf( -1 * (dp + *bias->get_row(0)[i] ) ) ) );
        }
    }*/
}

/*void Layer::back_prop(matrix* targets) {
    float** tar = targets->get_row(0);
    for(int i = 0; i < num_nodes; i++) {
        //float o = *outputs->get_row(0)[i];
        float o = *raw_outputs->get_row(0)[i]; //<---------- CHANGE FOR BATCH SZ
        float dbloc = (o * (1-o));

        if(targets != NULL) {
            dbloc *= (o - *tar[i]);
        } else {
            float** w = out_weights->get_row(i);
            float** ndb = next->del_bias->get_row(0);
            float db = 0.0;
            for(int j = 0; j < out_weights->num_cols; j++) {
                db += *ndb[j] * *w[j];
            }
            dbloc *= db;
        }

        *del_bias->get_row(0)[i] += dbloc;

        float** dw = in_del_weights->get_col(i);

        float** ins = inputs->get_row(0); //<---------- CHANGE FOR BATCH SZ

        for(int j = 0; j < in_del_weights->num_rows; j++) {
            *dw[j] += dbloc * *ins[j];
        }
    }
}*/

void Layer::back_prop(matrix* targets, int batch_sz) {
    if(targets != NULL) {
        elwise_subtract(outputs, targets, outputs);
    } else {
        transpose(out_weights, out_weightsT);
        mat_mul(out_weightsT, next->raw_outputs, outputs);
    }
    activate_prime(raw_outputs, raw_outputs, 0);
    elwise_mult(outputs, raw_outputs, raw_outputs);
    transpose(inputs, inputsT);
    mat_mul(raw_outputs, inputsT, in_del_weights);
    divide(in_del_weights, in_del_weights, batch_sz);
    sum_reduce_rows(raw_outputs, del_bias);
    divide(del_bias, del_bias, batch_sz);
    /* for output layer:
       1. outputs - targets -> outputs
       2. sigmoid_prime(raw_outputs) -> raw_outputs
       3. element_mul(outputs, raw_outputs) -> raw_outputs
       4. Transpose(inputs) -> inputsT
       5. mat_mul(raw_outs, inputsT) -> del_Weights / batch_sz
       6. sum_cols(raw_outputs) - > del_bias / batch_sz
       */

    /* for hidden layer:
       1. Transpose(out_weights) -> out_weightsT
       2. mat_mul(out_weightsT, next->raw_outputs) -> outputs
       3. sigmoid_prime(raw_outputs) -> raw_outtputs
       4. element_mul(outputs, raw_outputs) -> raw_outputs
       5. Transpose(inputs) -> inputsT
       6. mat_mul(raw_outs, inputsT)
       7. sum_cols(raw_outputs) - > del_bias / batch_sz
       */

}


void Layer::update(float learn_rate, int batch_size) {
    float** b = bias->get_row(0);
    float** db = del_bias->get_row(0);

    for(int i = 0; i < num_nodes; i++) {
        float** w = in_weights->get_col(i);
        float** dw = in_del_weights->get_col(i);

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


