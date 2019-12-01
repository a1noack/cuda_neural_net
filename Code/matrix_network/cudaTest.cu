#include "cuda_kernels.cu"

int main(int argc, char *argv[]) {
    int r1 = 1, c1 = 15, r2 = 15, c2 = 4;

    float m1dat[r1*c1] = {0};
    float m2dat[r2*c2] = {0};

    for(int i = 0; i < r1*c1; i++) {
        m1dat[i] = 4.;
    }

    for(int i = 0; i < r2*c2; i++) {
        m2dat[i] = 3.;
    }

    matrix *m1 = new matrix(r1, c1);
    matrix *m2 = new matrix(r2, c2);
    matrix *m3 = new matrix(r1, c2);

    m1->set_memory(m1dat);
    m2->set_memory(m2dat);
    m3->set_mem_zero();

    m1->move_to_device();
    m2->move_to_device();
    m3->move_to_device();

    m1->print();
    m2->print();
    m3->print();

    mat_mul(m1, m2, m3);
    
    m3->print();

    exit(0);
}
