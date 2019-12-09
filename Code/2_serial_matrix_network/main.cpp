#include "matrix.hpp"

int main() {
    int rows = 3;
    int cols = 4;

    float ndat[12] = {1,2,3,4,5,6,7,8,9,10,11,12};

    printf("New matrix create\n");
    matrix* n = new matrix(3, 4);

    printf("set mem\n");
    n->set_memory(ndat);

    n->print();

    printf("set zeroes\n");
    n->set_mem_zero();

    n->print();

    printf("randoms\n");
    n->set_mem_random();

    n->print();

    delete n;
}

