/* Definition of the network class. This class will hold all of the network layers and implement the various batching strategies and cost functions.
 */

#include "layers.hpp"

class Network {
    private:
        Layer** layers;
        int num_layers;
        matrix* predictions;
        void train();

    public:
        Network();
        void add_layer(Layer*);
        void load_data(float*);
        void train_CEL();
        void train_MSEL();

};



