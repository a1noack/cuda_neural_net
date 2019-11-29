#include "layer.hpp"

int main() {
    Layer* l1 = new Layer(2, input, NULL);
    Layer* l2 = new Layer(2, hidden, l1);
    Layer* l3 = new Layer(2, output, l2);
}
