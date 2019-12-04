#include "matrix.hpp"
#include <math.h>

enum layer_pos {input, hidden, output};

class Layer {
    protected:


    public:

        int num_nodes;
        layer_pos lp;

        Layer* prev;
        Layer* next;

        matrix* outputs;
        matrix* inputs;

        matrix* in_weights;
        matrix* out_weights;
        matrix* in_del_weights;
        matrix* out_del_weights;

        matrix* bias;
        matrix* del_bias;

        matrix* raw_outputs;

        Layer(int, layer_pos, Layer*, int);
        ~Layer();

        void set_next_layer(Layer*);
        void set_output(float* ins) { outputs->set_memory(ins); }
        void zero_grad();
        void move_to_device();
        void move_to_host();

        void forward_pass();
        void back_prop(float* targets);
        void update(float, int);

        //TESTING FUNCTIONS:
        //
        void print_outputs() { outputs->print(); }
        void print_inputs() { inputs->print(); }
        void print_in_weights() { in_weights->print(); }
        void print_out_weights() { out_weights->print(); }
        void print_in_del_W() { in_del_weights->print(); }
        void print_out_del_W() { out_del_weights->print(); }
        void print_bias() { bias->print(); }
        void print_del_bias() { del_bias->print(); }

        void print_layer();

        void set_weights(float* w) { out_weights->set_memory(w); }
        void set_bias(float* b) { bias->set_memory(b); }


        matrix* w_idx;
        matrix* n_idx;
};

float dot_prod(float*, float*, int);
float dot_prod(float**, float**, int);
float MSE(float*, float*, int);
float MSE(float**, float*, int);
float MSE(float**, float**, int);
