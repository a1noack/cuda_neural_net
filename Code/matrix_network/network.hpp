#include "layer.hpp"
#include "utils/dataset.hpp"

typedef enum {RELU, Sigmoid, Softmax} layer_type ;

class Network {
    private:
        int num_layers;
        Layer** layers;

    public:
        Network(int, int*, layer_type*);

        void train(int, int, float, float);
        void set_input(float*);
        void zero_grad();

        void forward();
        void back_prop(float*);
        void update_weights(float, int);

        float** get_predictions();
        int get_num_predictions();
};

float average_err(float*, int n);

void print_f_arr(float* f, int n);
