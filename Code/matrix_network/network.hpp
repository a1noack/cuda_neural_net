#include "layer.hpp"
typedef enum {RELU, Sigmoid, Softmax} layer_type ;

class Network {
    private:
        int num_layers;
        Layer** layers;

    public:
        Network(int, int*, layer_type*);

        void train(int, int, float, float);
        void set_input();
        void zero_grad();
};
