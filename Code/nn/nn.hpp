
#include "layer.hpp"

/* The network can be seen as a set of layers. Each layer contains a matrix of weights and a matrix of outputs */

/* --------------------------_TO DO----------------------------------------------------------
 * Could get rid of the private variables... Creating many un-needed functions
 * Need to create a data loader. That can define the number of output and input nodes.
 * May need to rewrite input and output layers as derived classes?
 * Sleep.
 * Destructor for layers!!!! Clean up memory leaks. Probably do this before scaling.
 * --------------------------------------------------------------------------------------- */

//class Layer;

class Network {
    private:
        int num_layers;
        Layer** layers;

    public:
        Network(int*, int);
        void connect();
        void forward_pass();
        void set_input(float*);
        void print_layers();
        void print_weights();
        void print_bias();
        float* get_output();
        void back_propogate(float*);
        void update_weights();
        void set_weights(float**);
        void set_bias(float**);
        ~Network();

};


