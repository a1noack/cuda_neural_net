#pragma once

#include <stdio.h>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <time.h>

#define RF_HI 0.99
#define RF_LO 0.01

/* This is the matrix class. It holds the data that represents the parameters in the network. This class will also handle the deivce to host and vice versa transfers. */

class matrix{
    private:

    public:
        float* host_data;
        float* device_data;

        int num_rows;
        int num_cols;
        int num_vals;
        int on_device;

        matrix(int, int);
        ~matrix();

        void move_to_device();
        void move_to_host();

        void set_memory(float*);

        void set_mem_zero();
        void set_mem_random();

        void print_dims();
        void print();

        void mat_copy_from(matrix*);
        void set_data_loader(float**);
};

