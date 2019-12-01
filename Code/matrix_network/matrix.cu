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

void matrix::move_to_device() {
    if(!on_device) {
        cudaMemcpy(device_data, host_data, num_vals * sizeof(float), cudaMemcpyHostToDevice);    
        on_device = 1;
    }
}

void matrix::move_to_host() {
    if(on_device) {
        cudaMemcpy(host_data, device_data, num_vals * sizeof(float), cudaMemcpyDeviceToHost);    
        on_device = 0;
    }
}

void matrix::set_memory(float* new_vals) {
    std::memcpy(host_data, new_vals, sizeof(float) * num_rows * num_cols);
}

void matrix::set_memory(float* new_vals, int new_rows, int new_cols) {
    num_rows = new_rows;
    num_cols = new_cols;
    num_vals = new_rows * new_cols;

    delete [] host_data;

    host_data = new float[num_vals];

    std::memcpy(host_data, new_vals, sizeof(float) * num_vals);
}

void matrix::set_mem_zero() {
    memset(host_data, 0, sizeof(float) * num_vals);
}

void matrix::set_mem_random() {
    for(int i = 0; i < num_vals; i++) {
        host_data[i] = RF_LO + (float) ( rand() / ( (float) (RAND_MAX / (RF_HI - RF_LO))));

    }
}

//Returns an array of float pointers that are pointer to the values in the row requested
float** matrix::get_row(int index) {
    float** row_at_index = new float*[num_cols];

    if(!on_device) {
        for(int i = 0; i < num_cols; i++) {
            row_at_index[i] = &host_data[(index * num_cols) + i];
        }
        return row_at_index;
    }
    else {
        float **d_row_at_index;
        for(int i = 0; i < num_cols; i++) {
            row_at_index[i] = &device_data[(index * num_cols) + i];
        }
        cudaMalloc(&d_row_at_index, num_cols * sizeof(float*));
        cudaMemcpy(d_row_at_index, row_at_index, num_cols * sizeof(float*), cudaMemcpyHostToDevice);
        return d_row_at_index;
    }

}

float** matrix::get_col(int index) {
    float** col_at_index = new float*[num_rows];
    int j = 0;

    if(!on_device) {
        for(int i = index; i < num_rows * num_cols; i+=num_cols) {
            col_at_index[j] = &host_data[i];
            j++;
        }
        return col_at_index;
    }
    else {
        float **d_col_at_index;
        for(int i = index; i < num_rows * num_cols; i+=num_cols) {
            col_at_index[j] = &device_data[i];
            j++;
        }
        cudaMalloc(&d_col_at_index, num_rows * sizeof(float*));
        cudaMemcpy(d_col_at_index, col_at_index, num_rows * sizeof(float*), cudaMemcpyHostToDevice);
        return d_col_at_index;
    }
}


float** matrix::get_all_data() {
    float** all_data = new float*[num_vals];

    for(int i = 0; i < num_vals; i++) {
        all_data[i] = &host_data[i];
    }
    return all_data;
}

void matrix::print() {
    if(!on_device) {
        printf("on host: \n");
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                printf("%.1f ", host_data[(i*num_cols) + j]);
            }
            printf("\n");
        }
    }
    else {
        float temp[num_vals];
        cudaMemcpy(temp, device_data, num_vals * sizeof(float), cudaMemcpyDeviceToHost);
        printf("on device: \n");
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < num_cols; j++) {
                printf("%.1f ", temp[(i*num_cols) + j]);
            }
            printf("\n");
        }
    }
}

