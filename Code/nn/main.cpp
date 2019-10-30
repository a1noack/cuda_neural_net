#include "nn.hpp"

int main() {
    srand(static_cast <unsigned> (time(NULL)));
    int layout[3] = {5,3,2};
    int n = 3;

    Network* my_net = new Network(layout, n);

    my_net->connect();

    int in_dat[5] = {0,0,0,1,1};

    my_net->set_input(in_dat);

    my_net->print_layers();
    my_net->print_weights();
    my_net->forward_pass();
    my_net->print_layers();
    my_net->print_weights();

    delete my_net;
}
