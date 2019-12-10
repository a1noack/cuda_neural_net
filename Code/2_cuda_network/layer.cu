#include "layer.hpp"
#include "cuda_kernels.cu"

// Layer constructor creates the layers, and links them to the next and previous layers in the network. Sincel the layers will hold pointers to weight parameters in and out, linking is crucial
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
}

//This function sets the pointers in the layers and connects it to the next layer
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

}

//This function zeros out the gradients in the layer
void Layer::zero_grad() {
    if(lp != output) {
        del_bias->set_mem_zero();
        out_del_weights->set_mem_zero();
    }
}

//This function handles the movement between host and device for the matrix members
void Layer::move_to_device() {
    if(inputs != NULL) {
        inputs->move_to_device();
        inputsT->move_to_device();
    }
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
        out_weightsT->move_to_device();
    }
    if(bias != NULL) {
        bias->move_to_device();
        del_bias->move_to_device();
    }
}

//This function handles device to host transfers of all the matrix members
void Layer::move_to_host() {
    if(inputs != NULL) {
        inputs->move_to_host();
        inputsT->move_to_host();
    }
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
        out_weightsT->move_to_host();
    }
    if(bias != NULL) {
        bias->move_to_host();
        del_bias->move_to_host();
    }
}

//This is the forward pass function. It computes the outputs for each layer given the weight and bias parameters attatched to it
void Layer::forward_pass() {
    if(lp != input) {
        mat_mul(inputs, in_weights, raw_outputs);
        add_bias(raw_outputs, bias);
        activate(raw_outputs, outputs, 0);
    }
}

//This function preforms the backwards pass, different from output layers and hidden layers. Targets must be passed for the output layer. Targets are set to NULL for the hidden layers.
void Layer::back_prop(matrix* targets, int batch_sz) {
    if(targets != NULL) {
        elwise_subtract(outputs, targets, outputs);
    } else {
        transpose(out_weights, out_weightsT);
        mat_mul(next->raw_outputs, out_weightsT, outputs);
    }
    activate_prime(raw_outputs, raw_outputs, 0);
    elwise_mult(outputs, raw_outputs, raw_outputs);
    transpose(inputs, inputsT);
    mat_mul(inputsT, raw_outputs, in_del_weights);
    sum_reduce_rows(raw_outputs, del_bias);
}

//This function handles the parameter updates for the network.
void Layer::update(float learn_rate, int batch_size) {
    update_cuda(in_weights, in_del_weights, learn_rate / (float)batch_size);
    update_cuda(bias, del_bias, learn_rate / (float)batch_size);
}

//Handy Function to print the layer parameters
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

Layer::~Layer() {
    if( outputs != NULL) delete outputs;
    if( inputs != NULL) delete inputs;
    if( out_weights != NULL) delete out_weights;
    if( in_weights != NULL) delete in_weights;
    if( in_del_weights != NULL) delete in_del_weights;
    if( out_del_weights != NULL) delete out_del_weights;
    if( bias != NULL) delete bias;
    if( del_bias != NULL) delete del_bias;
    if( raw_outputs != NULL) delete raw_outputs;
    if( out_weightsT != NULL) delete out_weightsT;
    if( inputsT != NULL) delete inputsT;
}

//This function is a CUDA kernel wrapper function to compute mean squared error
float MSE_mat_wrapper(matrix *y, matrix *yhat, matrix *result) {
    return MSE_mat(y, yhat, result);
}
