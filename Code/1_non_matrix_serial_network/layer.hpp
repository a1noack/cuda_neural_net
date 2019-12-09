#pragma once

#include <ctime>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <stdio.h>

#define r_HI 1.0
#define r_LO 0.0

class Layer {

    protected:
        Layer* prev_layer;
        Layer* next_layer;
        int num_nodes;
        float** weights;
        float* outputs;
        float* bias;

        float** del_weights;
        float* del_bias;

    public:
        //Layer(int) {};
        void set_output(float*);
        void set_next_layer(Layer* n) { next_layer = n; }
        void set_prev_layer(Layer* p) {prev_layer = p; }
        int get_num_nodes() { return num_nodes; }
        float* get_outputs() { return outputs; }
        float* get_bias() { return bias; }

        void connect_layers(Layer*);
        virtual void forward_pass(Layer*) = 0; //<------ Forward pass. Rename
        void print_lweights();

        virtual void back_prop_input(float*) = 0;
        virtual void back_prop() = 0;

        float** get_weights() { return weights; }
        float* get_del_bias() { return del_bias; }
        float** get_del_weights() { return del_weights; }

        void set_weights(float*);
        void set_bias(float*);
        void update(float, int);

        void zero_grad();
        void print_bias();
        virtual ~Layer() {};
};

float get_random_f();

float dot_prod(float*, float*, int);

float MSE(float*, float*, int);

void print_f_arr(float*, int);
