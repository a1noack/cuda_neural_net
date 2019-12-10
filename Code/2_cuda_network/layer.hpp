#include "matrix.hpp"
#include <math.h>

//Layer class. This holds all the parameters for the current layer and pointers to the next and previous layers, as well as weights

//enum to define layer position
enum layer_pos {input, hidden, output};

class Layer {
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
        matrix* out_weightsT;
        matrix* inputsT;
        Layer(int, layer_pos, Layer*, int);
        ~Layer();
        void set_next_layer(Layer*);
        void set_output(float* ins) { outputs->set_memory(ins); }
        void zero_grad();
        void move_to_device();
        void move_to_host();
        void forward_pass();
        void back_prop(matrix*, int);
        void update(float, int);
        void print_layer();
        void set_weights(float* w) { out_weights->set_memory(w); }
        void set_bias(float* b) { bias->set_memory(b); }
};

float MSE_mat_wrapper(matrix *, matrix *, matrix *);
