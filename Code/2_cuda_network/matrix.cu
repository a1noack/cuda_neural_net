#include "matrix.hpp"

//matrix constructor
matrix::matrix(int rows, int cols) {
    num_rows = rows;
    num_cols = cols;
    num_vals = rows * cols;
    on_device = 0;
    host_data = new float[num_vals];
    cudaMalloc(&device_data, num_vals * sizeof(float));
}

//matrix destructor
matrix::~matrix() {
    cudaFree(device_data);
    delete [] host_data;
}

//function to move the matrix to the GPU
void matrix::move_to_device() {
    if(!on_device) {
        cudaMemcpy(device_data, host_data, num_vals * sizeof(float), cudaMemcpyHostToDevice);
        on_device = 1;
    }
}

//Function to move the data from the GPU to the host
void matrix::move_to_host() {
    if(on_device) {
        cudaMemcpy(host_data, device_data, num_vals * sizeof(float), cudaMemcpyDeviceToHost);
        on_device = 0;
    }
}

//Function to set the values in the matrix. Handles GPU memory and host memory
void matrix::set_memory(float* new_vals) {
    if(!on_device) {
        std::memcpy(host_data, new_vals, sizeof(float) * num_rows * num_cols);
    } else {
        cudaMemcpy(device_data, new_vals, num_vals * sizeof(float), cudaMemcpyHostToDevice);
    }
}

//function to set all the memory in the matrix to zero
void matrix::set_mem_zero() {
    if(on_device) {
        cudaMemset(device_data, 0, sizeof(float) * num_vals);
    }
    else{
        memset(host_data, 0, sizeof(float) * num_vals);
    }
}

//function to set all the memory to random floats
void matrix::set_mem_random() {
    for(int i = 0; i < num_vals; i++) {
        host_data[i] = RF_LO + (float) ( rand() / ( (float) (RAND_MAX / (RF_HI - RF_LO))));

    }
}

//nice function to print the matrix
void matrix::print() {
    if(!on_device) {
        printf("(on host) \n");
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                printf("%.5f ", host_data[(i*num_cols) + j]);
            }
            printf("\n");
        }
    }
    else {
        float temp[num_vals];
        cudaMemcpy(temp, device_data, num_vals * sizeof(float), cudaMemcpyDeviceToHost);
        printf("(on device) \n");
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                printf("%.5f ", temp[(i*num_cols) + j]);
            }
            printf("\n");
        }
    }
}

//Function to print the matrix dimensions
void matrix::print_dims() {
    printf("%d x %d\n", num_rows, num_cols);
}

void matrix::mat_copy_from(matrix* a) {
    if(device_data != NULL && num_vals == a->num_vals) {

        cudaMemcpy(device_data, a->device_data, a->num_vals * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    if(host_data != NULL && num_vals == a->num_vals) {
        memcpy(host_data, a->host_data, sizeof(float) * num_vals);
    }
}

//Function to set the input data from the data loader
void matrix::set_data_loader(float** data) {
    float* data_1d = new float[num_cols * num_rows];

    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            data_1d[(i * num_cols) + j] = data[i][j];
        }
    }

    set_memory(data_1d);
}

