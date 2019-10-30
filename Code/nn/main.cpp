#include "nn.hpp"

int main() {
    srand(static_cast <unsigned> (time(NULL)));
    int layout[4] = {5,7,5,2};
    int n = 4;

    Network* my_net = new Network(layout, n);

    my_net->connect();

    int in_dat[5] = {0,0,0,1,1};

    my_net->set_input(in_dat);

    my_net->print_layers();
    my_net->print_weights();
    my_net->forward_pass();
    my_net->print_layers();
    my_net->print_weights();
    printf("\n\n");

    int out_dat[2] = {0,1};

    my_net->back_propogate(out_dat);
    my_net->update_weights();
    my_net->print_weights();
    delete my_net;
}
