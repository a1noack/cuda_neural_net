/* definition file for the matrix class. This class stores the values and manages the memory between host and device. Functions set() and replace() will update the data on both host and device. */
#pragma once

#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>

#define LO_r 0.01
#define HI_r 0.99

class matrix {
	private:
        bool dev_alloc;
        bool host_alloc;

        void alloc_dev_mem();
        void alloc_host_mem();

    public:
        int dim_x;
        int dim_y;

        float* dev_data;
        float* host_data;

        matrix();
        matrix(int x, int y);

        void mem_alloc();
        void mem_alloc_if_null();

        void copy_host_to_dev();
        void copy_dev_to_host();

        void set_mem(float*, int, int);
        void set_mem_zero(int, int);
        void set_mem_random(int, int);
        void update_mem(float*);
        void print();

        ~matrix();

        //======== TESTING FUNCTIONS =========
        void pst_vals();
        void add_one();
//        float& operator[](const int index);
//        const float& operator[](const int index) const;
};


