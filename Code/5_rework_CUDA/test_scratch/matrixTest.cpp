#include "matrix.hpp"

int main() {
    int rows = 3;
    int cols = 4;

    float ndat[12] = {1,2,3,4,5,6,7,8,9,10,11,12};

    printf("New matrix created\n");
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

    n->move_to_device();
    printf("testing moving to device\n");
    n->print();

    n->get_row(1);
    n->get_col(1);

    delete n;
}

