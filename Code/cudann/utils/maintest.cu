#include "matrix.hpp"

#include <stdio.h>
#include <stdlib.h>

int main() {
    matrix *a = new matrix(64,64);
    a->mem_alloc();
    a->pst_vals();
    //a->print();
    a->copy_host_to_dev();
    a->add_one();
    a->copy_dev_to_host();
    a->print();
    delete a;
}

