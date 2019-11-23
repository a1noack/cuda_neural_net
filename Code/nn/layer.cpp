#include "layer.hpp"

/* Again, stupid. Sets all the outputs of the first (input) layer, as the training data example. */
void Layer::set_output(float* data) {
    for(int i = 0; i < num_nodes; i++) {
        outputs[i] = data[i];
    }
}

/* Layer constructor. Initializes many data members. What it cannot it will set to NULL, hopefully avoid shitty behavior */
Layer::Layer(int numNodes) {
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

    printf("layer Created %d nodes\n", this->num_nodes);
}

/* SIGMOID ACTIVATION!!! */
void Layer::compute_outputs(Layer* prev) {
    for(int i = 0; i < num_nodes; i++) {
        float dp = dot_prod(weights[i], prev->get_outputs(), num_nodes);
        outputs[i] = (1 / (1+ expf( -1* (dp + bias[i]) ) ) );
    }
}

/* Clearly a incomplete destructor */
Layer::~Layer() {
}

/* function to connect layers with weights and randomize weights, each layer owns its input edges! */
void Layer::connect_layers(Layer* prev) {
    prev->set_next_layer(this);
    this->set_prev_layer(prev);
    this->weights = new float*[num_nodes];
    this->del_weights = new float*[num_nodes];

    for(int i = 0; i < this->num_nodes; i++) {
        float* wl = new float[prev->get_num_nodes()];

        for (int j = 0; j < prev->get_num_nodes(); j++) {
            wl[j] = get_random_f();
        }

        weights[i] = wl;
    }
    printf("Layer Connected\n");
}

/* The good shit. Back propogate the error on just the input layer. */
void Layer::back_prop_input(float* targets) {
    int num_weights = prev_layer->get_num_nodes();

    for(int i = 0; i < num_nodes; i++) {
        float o = outputs[i];
        //printf("tar[i] %f\n", targets[i]);
        //printf("term1 = %f ", o-targets[i]);
        //printf("term2 = %f ", o * (1.0-o));
        del_bias[i] = (o - targets[i]) * (o * (1.0 - o));
        //printf("delb = %f\n", del_bias[i]);

        float* dw = new float[num_weights];
        float* po = prev_layer->get_outputs();

        for(int j = 0; j < num_weights; j++) {

            dw[j] = del_bias[i] * po[j];
            //printf("dw = %f, (bd = %f, pw = %f\n", dw[j], del_bias[i], po[j]);
        }

        del_weights[i] = dw;
    }
    //printf("\n");

}

/* the really good shit. Back propgates the error for the hidden layers */
void Layer::back_prop() {
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
        del_bias[i] = db * (o * (1.0-o));
        //printf("fdb:::%f\n", del_bias[i]);
        float* dw = new float[num_nodes];

        float* po = prev_layer->get_outputs();

        for(int k = 0; k < num_nodes; k++) {
            dw[k] = del_bias[i] * po[k];
          //  printf("dw = %f, db = %f, po = %f\n", dw[k], del_bias[i], po[k]);
        }

        del_weights[i] = dw;
    }
}

/* update weights on each layer */
void Layer::update() {
    double learn_rate = 0.01; //<---------------- Maybe put this somewhere else

    int num_weights = prev_layer->get_num_nodes(); //<------- BAD

    for(int i = 0; i < num_nodes; i++) {
        float* w = weights[i];
        float* dw = del_weights[i];

        for(int j = 0; j < num_weights; j++) {
            w[j] = w[j] - (learn_rate * dw[j]);
        }

        bias[i] = bias[i] - (learn_rate * del_bias[i]);
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

    return ((float) 1 / (float) n) * s;
}