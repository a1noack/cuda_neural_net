#include "layer.hpp"

//Layer class constructor. Starts with lots of NULL assignments. Some layers depending on the position do not have some parameters, and they will be filled in if needed.
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

    //here is where we set the current layer's previous layer field, if that parameter is not NULL (input layers do not have a previous layer
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

//Sets the current layer's next layer. Separate from the constructor. Layers share weights (in vs out) so we need to set and allocate pointers somewhere.
void Layer::set_next_layer(Layer* next_layer) {

    next = next_layer;
    out_weights = new matrix(num_nodes, next->num_nodes);
    out_weights->set_mem_random();

    next->in_weights = out_weights;
    next->inputs = outputs;

    out_del_weights = new matrix(num_nodes, next->num_nodes);
    out_del_weights->set_mem_zero();
    next->in_del_weights = out_del_weights;

}

//Zeroes out the gradients in each layer
void Layer::zero_grad() {
    if(del_bias != NULL) {
        del_bias->set_mem_zero();
    }
    if(out_del_weights != NULL) {
        out_del_weights->set_mem_zero();
    }
}

//Forward pass function computes the predicted outputs of the network with the current weights and biases
void Layer::forward_pass() {
    if(lp != input) {

        for(int i = 0; i < num_nodes; i++) {

            assert(in_weights->num_rows == inputs->num_cols);
            //<---------- COLUMN CALL
            float** in_w = in_weights->get_col(i);
            float** in = inputs->get_row(0);
            float** b_bias = bias->get_row(0);

            float dp = dot_prod(in_w, in, in_weights->num_rows);
            float** o_outs = outputs->get_row(0);

            *o_outs[i] = ( 1 / ( 1 + expf( -1 * (dp + *b_bias[i] ) ) ) );

            delete [] in_w;
            delete [] in;
            delete [] b_bias;
            delete [] o_outs;
        }
    }
}

//Back prop function. Computes the gradients (weights and biases) for each layer. Back prop is done differently between the output layer and hidden layers. Outputs the expected outputs(targets) are passed in, hidden layers a NULL pointer is passed in since we do not use the targets to compute the gradients in the hidden layers.
void Layer::back_prop(float* targets) {
    for(int i = 0; i < num_nodes; i++) {
        assert(num_nodes == outputs->num_cols);
        float** o_outs = outputs->get_row(0);

        float o = *o_outs[i];
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

            delete [] w;
            delete [] ndb;
        }
        delete [] o_outs;

        float** cdb = del_bias->get_row(0);
        *cdb[i] += dbloc;
        delete [] cdb;

        float** dw = in_del_weights->get_col(i);

        float** ins = inputs->get_row(0);

        assert(in_del_weights->num_rows == inputs->num_cols);

        for(int j = 0; j < in_del_weights->num_rows; j++) {
            *dw[j] += dbloc * *ins[j];
        }

        delete [] dw;
        delete [] ins;
    }
}

// Update function. Updates the layer weights and biases given the gradients computed in back_prop. Learning rate scales the update, batch size helps compute an aggregate gradient
void Layer::update(float learn_rate, int batch_size) {
    float** b = bias->get_row(0);
    float** db = del_bias->get_row(0);

    assert(num_nodes == in_weights->num_cols);

    for(int i = 0; i < num_nodes; i++) {
        float** w = in_weights->get_col(i);
        float** dw = in_del_weights->get_col(i);

        assert(prev->num_nodes == in_weights->num_rows);

        for(int j = 0; j < prev->num_nodes; j++) {
            *w[j] = *w[j] - (learn_rate * (*dw[j] / batch_size) );
        }

        *b[i] = *b[i] - (learn_rate * (*db[i] / (float)batch_size) );

        delete [] w;
        delete [] dw;
    }

    delete [] b;
    delete [] db;
}

//messy function to print all the layers
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

//iterative dot product between two array floats
float dot_prod(float** x, float** y, int num) {
    float dp = 0.0;

    for(int i = 0; i < num; i++) {
        dp += *x[i] * *y[i];
    }
    return dp;
}

//Mean square error computation
float MSE(float* v1, float* v2, int num) {
    float s = 0.0;

    for(int i = 0; i < num; i++) {
        s += pow( (double) v1[i] - v2[i], 2);
    }

    return ( (float) 1 / (float) (num * 2) ) * s;
}

