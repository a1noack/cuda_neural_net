/* this is the implementation for the layer class. There should be a virtual parent class and then the different types of layers should be created form there. */

#include "matrix.hpp"
#include <stdio.h>
#include <stdlib.h>

class Layer {
    private:
        Layer* next;
        Layer* previous;
        char* name;
        matrix* weights;
        matrix* dweights
    public:
        Layer() {};
        ~Layer() {};
        virtual void forward() = 0;
        virtual void backprop() = 0;
};

class LinearLayer {

