/* definition file for the matrix class. This class stores the values for each layer and convolutional kernels. It needs to keep the device and host pointers*/
#pragma once

#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

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

        void pst_vals();
        void add_one();
        void print();
        ~matrix();
//        float& operator[](const int index);
//        const float& operator[](const int index) const;
};


