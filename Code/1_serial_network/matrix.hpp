#include <stdio.h>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <time.h>

#define RF_HI 0.99
#define RF_LO 0.01

/* This the matrix class that essentially holds all of the data for each parameter in the network. Dimensions are stored, including acces functions to rows and columns in the matrix. */
class matrix{
    public:
        int num_rows;
        float* host_data;
        int num_cols;
        int num_vals;

        matrix(int, int);
        ~matrix();

        void set_memory(float*);

        void set_mem_zero();
        void set_mem_random();

        float** get_row(int);
        float** get_col(int);

        void print();
};

