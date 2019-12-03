#include "cuda_kernels.cu"
// i don't need to include matrix.hpp because
// it is included in cuda_kernels.cu

int main(int argc, char *argv[]) {
    int r1 = 1, c1 = 15, r2 = 15, c2 = 4;

    float m1dat[r1*c1] = {0};
    float m2dat[r2*c2] = {0};
    float m3graddata[r1*c2] = {0};
    float biasdata[1*c2] = {0};

    for(int i = 0; i < r1*c1; i++) {
        m1dat[i] = .1;
    }

    for(int i = 0; i < r2*c2; i++) {
        m2dat[i] = .5;
    }

    for(int i = 0; i < 1*c2; i++) {
        biasdata[i] = i;
    }

    for(int i = 0; i < r1*c2; i++) {
        m3graddata[i] = .1;
    }

    matrix *m1 = new matrix(r1, c1);
    matrix *m2 = new matrix(r2, c2);
    matrix *m3 = new matrix(r1, c2);
    matrix *m3grad = new matrix(r1, c2);
    matrix *bias = new matrix(1, c2);

    m1->set_memory(m1dat);
    m2->set_memory(m2dat);
    m3grad->set_memory(m3graddata);
    bias->set_memory(biasdata);
    m3->set_mem_zero();

    m1->move_to_device();
    m2->move_to_device();
    m3->move_to_device();
    m3grad->move_to_device();
    bias->move_to_device();

    m1->print();
    m2->print();

    mat_mul(m1, m2, m3);
    m3->print();
    
    add_bias(m3, bias);
    m3->print();

    activate(m3, 0);
    m3->print();

    update(m3, m3grad, 1.);
    m3->print();

    printf("\n");
    elwise_subtract(m3grad, m3, m3grad);
    m3grad->print();

    elwise_mult(m3, m3grad, m3);
    m3->print();

    exit(0);
}
