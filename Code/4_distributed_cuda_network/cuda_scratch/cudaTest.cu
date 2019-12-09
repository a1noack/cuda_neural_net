#include "cuda_kernels.cu"
// i don't need to include matrix.hpp because
// it is included in cuda_kernels.cu

int main(int argc, char *argv[]) {
    int r1 = 20, c1 = 18, r2 = 18, c2 = 6;

    float m1dat[r1*c1] = {0};
    float m2dat[r2*c2] = {0};
    float m3graddata[r1*c2] = {0};
    float biasdata[1*c2] = {0};
    float biasdata2[1*c2] = {0};

    for(int i = 0; i < r1*c1; i++) {
        m1dat[i] = .1 * i;
    }

    for(int i = 0; i < r2*c2; i++) {
        m2dat[i] = .5;
    }

    for(int i = 0; i < 1*c2; i++) {
        biasdata[i] = 3.;
    }

    for(int i = 0; i < r1*c2; i++) {
        m3graddata[i] = .1;
    }

    for(int i = 0; i < 1*c2; i++) {
        biasdata2[i] = i*1;
    }

    matrix *m1 = new matrix(r1, c1);
    matrix *m1T = new matrix(c1, r1);
    matrix *m2 = new matrix(r2, c2);
    matrix *m3 = new matrix(r1, c2);
    matrix *m3grad = new matrix(r1, c2);
    matrix *bias = new matrix(1, c2);
    matrix *bias2 = new matrix(1, c2);

    m1->set_memory(m1dat);
    m1T->set_mem_zero();
    m2->set_memory(m2dat);
    m3grad->set_memory(m3graddata);
    bias->set_memory(biasdata);
    bias2->set_memory(biasdata2);
    m3->set_mem_zero();

    m1->move_to_device();
    m1T->move_to_device();
    m2->move_to_device();
    m3->move_to_device();
    m3grad->move_to_device();
    bias->move_to_device();
    bias2->move_to_device();

    printf("\nm3grad: "); m3grad->print();

    printf("\nm1: "); m1->print();
    transpose(m1, m1T);
    printf("\nm1T: "); m1T->print();

    printf("\nm2: ");
    m2->print();
    
    printf("\nm1 x m2 = m3: ");
    mat_mul(m1, m2, m3);
    
    printf("\nm3: ");
    m3->print();
    
    printf("\nm3 + bias: ");
    add_bias(m3, bias);
    m3->print();

    printf("\nsigmoid(m3): ");
    activate(m3, m3, 0);
    m3->print();

    printf("\nm3grad: "); m3grad->print();
    printf("\nm3 - m3grad = m3: ");
    update(m3, m3grad, 1.);
    m3->print();

    printf("\nm3grad - m3: ");
    elwise_subtract(m3grad, m3, m3grad);
    m3grad->print();

    printf("\nm3 * m3grad: ");
    elwise_mult(m3, m3grad, m3);
    m3->print();

    printf("\nm3 / 1000.: ");
    divide(m3, m3, 1000.);
    m3->print();

    printf("\nsigmoid_prime(m3): ");
    activate_prime(m3, m3, 0);
    m3->print();
    
    printf("\nsum_row_reduce: ");
    sum_reduce_rows(m3, m3);
    m3->print();

    printf("\nbias: ");
    bias->print();
    
    printf("\nbias2: ");
    bias2->print();

    printf("\nMSE(bias, bias2): ");
    float mse = MSE(bias, bias2, bias);
    printf("\n%f\n");
    
    exit(0);
}
