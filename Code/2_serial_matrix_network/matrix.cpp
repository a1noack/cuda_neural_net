#include "matrix.hpp"

//matrix constructor
matrix::matrix(int rows, int cols) {
    num_rows = rows;
    num_cols = cols;
    num_vals = rows * cols;

    host_data = new float[num_vals];

}

//matrix destructor
matrix::~matrix() {
    delete [] host_data;
}

//Function to set the values in the matrix.
void matrix::set_memory(float* new_vals) {
    std::memcpy(host_data, new_vals, sizeof(float) * num_rows * num_cols);
}

// Function to set all the matrix memory to zero
void matrix::set_mem_zero() {
    memset(host_data, 0, sizeof(float) * num_vals);
}

// Function to set random values in the matrix
void matrix::set_mem_random() {
    srand(time(NULL));
    for(int i = 0; i < num_vals; i++) {
        host_data[i] = RF_LO + (float) ( rand() / ( (float) (RAND_MAX / (RF_HI - RF_LO))));

    }
}

//Returns an array of float pointers that are pointer to the values in the row requested
float** matrix::get_row(int index) {
    float** row_at_index = new float*[num_cols];

    for(int i = 0; i < num_cols; i++) {
        row_at_index[i] = &host_data[(index * num_cols) + i];
    }

    return row_at_index;
}

//Returns an array of float pointer that are pointers to the values in the column requested
float** matrix::get_col(int index) {
    float** col_at_index = new float*[num_rows];
    int j = 0;
    for(int i = index; i < num_rows * num_cols; i+=num_cols) {
        col_at_index[j] = &host_data[i];
        j++;
    }
    return col_at_index;
}

//debug function to print the data in the matrix. Nice and ordered.
void matrix::print() {
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            printf("%f ", host_data[(i*num_cols) + j]);
        }
        printf("\n");
    }
}

