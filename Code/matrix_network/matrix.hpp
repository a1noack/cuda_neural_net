#pragma once

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
        float* host_data;
        float* device_data;

        int num_rows;
        int num_cols;
        int num_vals;
        int on_device;

        matrix(int, int); //matrix constrructor takes two dims
        ~matrix();

        void move_to_device();
        void move_to_host();

        void set_memory(float*);
        void set_memory(float*, int, int);

        void set_mem_zero();
        void set_mem_random();

        float** get_row(int);
        float** get_col(int);
        float** get_all_data();

        void print_dims();
        void print();

        void mat_copy_from(matrix*);
};

