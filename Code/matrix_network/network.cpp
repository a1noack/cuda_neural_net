#include "network.hpp"

Network::Network(int num_l, int* node_list, layer_type* lt, int num_batches) {
    num_layers = num_l;

    layers = new Layer*[num_layers];

    Layer* last_layer = NULL;

    for(int i = 0; i < num_layers; i++) {
        Layer* nl = NULL;

        layer_pos position = hidden;

        if(i == 0) {
            position = input;
        } else if (i == num_layers - 1) {
            position = output;
        }

        switch(lt[i]) {
            case(RELU): nl = new Layer(node_list[i], position, last_layer, num_batches); break;
            case(Sigmoid): nl = new Layer(node_list[i], position, last_layer, num_batches); break;
            case(Softmax): nl = new Layer(node_list[i], position, last_layer, num_batches); break;
        }

        last_layer = nl;
        layers[i] = nl;
    }
}

void Network::train(int num_epochs, int batch_size, float learn_rate, float min_err) {
    float cur_error = 0.0;

    int  cur_epoch, cur_batch = 0;

    char* file_name = "../data/data_n1000_m5_mu1.5.csv";

    Dataset d(file_name, batch_size);

    int total_samples = d.n;
    int num_batches = d.n / batch_size;

    for(int i = 0; i < num_epochs; i++) {
        printf("Epoch #%d, Current Error: %f\n", i + 1, cur_error);
        d.shuffle_sample_order();
        cur_epoch = i+1;

        for(int j = 0; j < num_batches; j++) {
            d.load_next_batch();
            this->zero_grad();
            float errors[batch_size];

            for(int k = 0; k < batch_size; k++) {
                this->set_input(d.batch_x[k]);
                this->forward();
                errors[k] = MSE(this->get_predictions(), d.batch_y[k], this->get_num_predictions());
                this->back_prop(d.batch_y[k]);
            }

            cur_batch = j+1;

            cur_error = average_err(errors, batch_size);

            if(cur_error <= min_err) { goto done_training; }
            this->update_weights(learn_rate, batch_size);
        }
    }

done_training:
    printf("Training finished at Epoch:%d, Batch%d, Error: %f\n", cur_epoch, cur_batch, cur_error);
}



void Network::set_input(float* ins) {
    layers[0]->set_output(ins);
}

void Network::zero_grad() {
    for(int i = 0; i < num_layers - 1; i++) {
        layers[i]->zero_grad();
    }
}

void Network::forward() {
    for(int i = 0; i < num_layers; i++) {
        layers[i]->forward_pass();
    }
}

void Network::back_prop(matrix* targets) {
    matrix* nts = targets;
    for(int i = num_layers - 1; i > 0; i--) {
        layers[i]->back_prop(nts);
        nts = NULL;
    }
}

void Network::update_weights(float learn_rate, int batch_size) {
    for(int i = num_layers - 1; i > 0; i--) {
        layers[i]->update(learn_rate, batch_size);
    }
}

float** Network::get_predictions() {
    return layers[num_layers - 1]->outputs->get_row(0);
}

int Network::get_num_predictions() {
    return layers[num_layers - 1]->num_nodes;
}

float average_err(float* errors, int num) {
    if(num == 1) {
        return errors[0];
    }

    float sum_errs = 0.0;
    for(int i = 0; i < num; i++) {
        sum_errs += errors[i];
    }
    return sum_errs / num;
}


void print_f_arr(float* f, int n) {
    for(int i = 0; i < n; i++) {
        printf("%f ", f[i]);
    }
    printf("\n");
}

