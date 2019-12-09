#include "matrix.hpp"

// constructor for a multidim container
matrix::matrix(int x, int y) {
    // number of rows is x dim, number of cols is y dim
    dim_x = x;
    dim_y = y;

    dev_alloc = false;
    host_alloc = false;
    mem_alloc();
}

// Constructor for a general container
matrix::matrix() {
    dim_x = 1;
    dim_y = 1;

    dev_alloc = false;
    host_alloc = false;

}

// allocates memory on GPU if not allocated
void matrix::alloc_dev_mem() {
    if(!dev_alloc) {
        dev_data = nullptr;
        cudaMalloc(&dev_data, dim_x * dim_y * sizeof(float));
        dev_alloc = true;
    }
}

// allocates memory on Host if not allocated
void matrix::alloc_host_mem() {
    if(!host_alloc) {
        host_data = new float[dim_x * dim_y];
        host_alloc = true;
    }
}

// function to allocate memory on both host and device
void matrix::mem_alloc() {
    alloc_host_mem();
    alloc_dev_mem();
}

// function to move memory from the host on to the device
void matrix::copy_host_to_dev() {
    if(dev_alloc && host_alloc) {
        cudaMemcpy(dev_data, host_data, dim_x * dim_y *sizeof(float), cudaMemcpyHostToDevice);
    }
}

// function to move device memory back to host
void matrix::copy_dev_to_host() {
    if(dev_alloc && host_alloc) {
        cudaMemcpy(host_data, dev_data, dim_x * dim_y *sizeof(float), cudaMemcpyDeviceToHost);
    }
}
// function will clear the device and host memory set both with a new array
void matrix::set_mem(float* new_vals, int x, int y) {
    if(host_alloc) {
        delete host_data;
        host_alloc = false;
    }

    if(dev_alloc) {
        cudaFree(dev_data);
        dev_alloc = false;
    }
    dim_x = x;
    dim_y = y;
    mem_alloc();
    memcpy(host_data, new_vals, sizeof(float) * (x*y));
    copy_host_to_dev();
}

//function to set memory to all zeroes
void matrix::set_mem_zero(int x, int y) {
    float* vals = (float*) calloc((x*y), sizeof(float));
    if(x == this->dim_x && y == this->dim_y) {
       this->update_mem(vals);
    } else {
       this->set_mem(vals, x, y);
    }
}

void matrix::set_mem_random(int x, int y) {
    int size = x * y;
    float* vals = new float[size];
    for(int i = 0; i < size; i++) {
        vals[i] = ((float) rand() / (float) RAND_MAX) * (HI_r - LO_r) - LO_r;
    }
    this->set_mem(vals, x, y);
    copy_host_to_dev();
}


// function to update memory if the memory is the same size.
// THIS FUNCTION WILL NOT CHECK MEMORY SIZE must check before.
void matrix::update_mem(float* new_vals) {
    memcpy(host_data, new_vals, dim_x * dim_y);
    copy_host_to_dev();
}

matrix::~matrix() {
    delete host_data;
    cudaFree(dev_data);
}

void matrix::print() {
    for(int i = 0; i < dim_x; i++) {
        for(int j = 0; j < dim_y; j++) {
            printf("%f ", host_data[i * dim_y + j]);
        }
        printf("\n");
    }

}

float* matrix::get_row(int idx) {
    float* row = new float[dim_y];
    memcpy(row, &host_data[idx*dim_y], sizeof(float) * dim_y);

    return row;
}

float* matrix::get_col(int idx) {
    float* col = new float[dim_x];
    int j = 0;
    for(int i = idx; i < dim_x * dim_y; i+=dim_y) {
        col[j] = host_data[i];
        j++;
    }
    return col;
}

// Stuff for matrix padding...

void matrix::pad_columns(int num_cols) {
    int num_to_pad = num_cols - dim_x;
    float* new_vals = (float*) calloc(num_cols * dim_y, sizeof(float));
    int j = 0;
    int i = 0;
    for(i = 0; i < num_cols * dim_y; i+=num_cols) {
        for(j = 0; j < num_to_pad; j++) {
            new_vals[(i * num_cols) + j] = host_data[(i*dim_x) + j];
        }
    }
    this->set_mem(new_vals, num_cols, dim_y);
}

void matrix::pad_rows(int num_rows) {
    int num_to_pad = num_rows - dim_y;
    float* new_vals = (float*) calloc(num_rows * dim_x, sizeof (float));
    memcpy(new_vals, host_data, dim_x * dim_y);
    this->set_mem(new_vals, dim_x, num_rows);
}


//===== ALL FUNCTIONS AFTER FOR TESTING ====
void matrix::pst_vals() {
    float a[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int itr = 0;
    for(int i = 0; i < dim_y; i++) {
        for(int j = 0; j < dim_x; j++) {
            host_data[i * dim_x + j] = a[itr];
            itr++;
        }
    }
}
__global__ void add_one_vec(float* dev_mem, int size) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Size: %d\n", size);
    printf("blockDimx: %d\n", blockDim.x);
    printf("blockIdxx: %d\n", blockIdx.x);
    printf("threadIdxx: %d\n", threadIdx.x);
    //int lcnt = 0;
    printf("TID: %d\n", tid);
    dev_mem[tid] += 1;
}

void matrix::add_one() {

    dim3 block_size(2,2);
    dim3 num_blocks( 32, 32);

    //add_one_vec<<<num_blocks, block_size>>>(dev_data, dim_x*dim_y);

    add_one_vec<<<32,32>>>(dev_data, dim_x*dim_y);
}




