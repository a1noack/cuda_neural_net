#include <stdio.h>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <time.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>

#define RF_HI 0.99
#define RF_LO 0.01

class matrix{
    private:


    public:
        int num_rows;
        float* host_data;
        int num_cols;
        int num_vals;

        matrix(int, int); //matrix constrructor takes two dims
        ~matrix();

        void set_memory(float*);
        void set_memory(float*, int, int);

        void set_mem_zero();
        void set_mem_random();

        float** get_row(int);
        float** get_col(int);
        float** get_all_data();

        void print();
};

