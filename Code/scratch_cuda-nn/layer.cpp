#include "layer.hpp"

/* Again, stupid. Sets all the outputs of the first (input) layer, as the training data example. */
void Layer::set_output(float* data) {
    for(int i = 0; i < num_nodes; i++) {
        outputs[i] = data[i];
    }
}



/* SIGMOID ACTIVATION!!! */


/* Clearly a incomplete destructor
Layer::~Layer() {
}
*/
/* function to connect layers with weights and randomize weights, each layer owns its input edges! */
void Layer::connect_layers(Layer* prev) {
    prev->set_next_layer(this);
    this->set_prev_layer(prev);
    this->weights = new float*[num_nodes];
    this->del_weights = new float*[num_nodes];

    for(int i = 0; i < this->num_nodes; i++) {
        float* wl = new float[prev->get_num_nodes()];
        float* dwl = new float[prev->get_num_nodes()];
        for (int j = 0; j < prev->get_num_nodes(); j++) {
            wl[j] = get_random_f();
            dwl[j] = 0.0;
        }

        weights[i] = wl;
        del_weights[i] = dwl;
    }
    printf("Layer Connected\n");
}



/* update weights on each layer */
void Layer::update(float learn_rate, int batch_size) {
    //double learn_rate = 0.01; //<---------------- Maybe put this somewhere else

/*    int num_weights = prev_layer->get_num_nodes(); //<------- BAD


    for(int i = 0; i < num_nodes; i++) {
        float* w = weights[i];
        float* dw = del_weights[i];

        for(int j = 0; j < num_weights; j++) {
            w[j] = w[j] - (learn_rate * (dw[j] / batch_size));
        }

        bias[i] = bias[i] - (learn_rate * (del_bias[i] / (float) batch_size));
    }
    */

    int num_weights = this->get_num_nodes() * prev_layer->get_num_nodes();
    float* w = weights->host_data;
    float* dw = del_weights->host_data;
    float* b = bias->host_data;
    float* db = del_bias->host_data;

    for(int i = 0; i < num_weights; i++) {
        w[i] = w[i] -(learn_rate * (dw[i] / batch_size));
        if(i < num_nodes) {
            b[i] = b[i] - (learn_rate * (db[i] / batch_size));
        }
    }

}


/* testing function, prints each weight */
void Layer::print_lweights() {
    for(int i = 0; i <num_nodes; i++) {
        float* f = weights[i];
        for(int j = 0; j < prev_layer->get_num_nodes(); j++) {
            printf("%f ", f[j]);
        }
        printf("\n");
    }
    printf("\n");
}

void Layer::print_bias() {
	for(int i = 0; i < num_nodes; i++) {
		printf("%f ", bias[i]);
	}
	printf("\n");
}
void Layer::set_weights(float* n_weights) {
	int n_itr = 0;
	for(int i = 0; i < num_nodes; i++) {
		float* cur_w = weights[i];
		for(int j = 0; j < num_nodes; j++) {
			cur_w[j] = n_weights[n_itr];
			n_itr++;
		}
	}
}

void Layer::set_bias(float* n_bias) {
	for(int i = 0; i < num_nodes; i++) {
		bias[i] = n_bias[i];
	}
}

void Layer::zero_grad() {

     int num_weights = prev_layer->get_num_nodes(); //<------- BAD

    for(int i = 0; i < num_nodes; i++) {
        del_bias[i] = 0.0;
        float* dw = del_weights[i];
        for(int j = 0; j < num_weights; j++) {
            dw[j] = 0.0;
        }
    }
    //print_f_arr(del_bias, num_nodes);
}
/*************************************************************************************
	Accessory functions:
 ************************************************************************************/

/* Get a random float */
float get_random_f() {

    return r_LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(r_HI-r_LO)));
}

/* Calc dot product of two arrays of floats */
float dot_prod(float* x, float* y, int num) {

    float dp = 0.0;
    for(int i = 0; i < num; i++) {
        dp += x[i] * y[i];
    }

    return dp;
}

void print_f_arr(float* a, int n) {
	for(int i = 0; i < n; i++) {
		printf("%f ", a[i]);
	}
	printf("\n");
}
/* calculate mean squared error of two float arrays */
float MSE(float* v1, float* v2, int n) {
	//printf("Finding MSE of the following arrays of len %d: \n", n);

	//print_f_arr(v1, n);
	//print_f_arr(v2, n);

    float s = 0.0;
    for(int i = 0; i < n; i++) {
        s += pow(static_cast <double> (v1[i] - v2[i]), 2);
    }

    //printf("got to ret\n");
    return ((float) 1 / (float) n) * s;
}

