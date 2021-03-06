#include <stdio.h>
#include "matrix.hpp"

// make sure T is bigger than the number of neurons in the largest layer and a multiple of 32
#define T 1024

// a variety of sum reduction methods (https://jeewhanchoi.github.io/uocis631f19/lecture12.pptx)
#define naive_reduce { \
    for(int s = 1; s < blockDim.x; s*=2) { \
        if(l_tid % (2 * s) == 0) { \
            shared[l_tid] = shared[l_tid] + shared[l_tid + s]; \
        } __syncthreads(); }}

#define nondivergent_branch_reduce {\
    int idx; \
    for(int s = 1; s < blockDim.x; s*=2) { \
        idx = 2 * s * l_tid; \
        if(idx < blockDim.x) { \
            shared[l_tid] += shared[l_tid + s]; \
        } __syncthreads(); }}

#define seq_addressing_reduce {\
    for(int s = blockDim.x / 2; s > 0; s>>=1) { \
        if(l_tid < s) { \
            shared[l_tid] += shared[l_tid + s]; \
        } __syncthreads(); }}

#define loop_unrolled_reduce {\
    for(unsigned int s = blockDim.x / 2; s > 32; s>>=1) { \
        if(l_tid < s) { \
            shared[l_tid] += shared[l_tid + s]; \
        } __syncthreads();} \
    if(l_tid < 32) { \
        shared[l_tid] += shared[l_tid + 32]; \
        shared[l_tid] += shared[l_tid + 16]; \
        shared[l_tid] += shared[l_tid + 8]; \
        shared[l_tid] += shared[l_tid + 4]; \
        shared[l_tid] += shared[l_tid + 2]; \
        shared[l_tid] += shared[l_tid + 1]; }}

#define _reduce_ loop_unrolled_reduce


/*Cuda kernels*/

__global__ void _sum_reduce(float *arr, float *result, int n) {
    /**
     * Sums across all elements in an array arr and places result at &result[0].
     * n is the number of elements in arr.
     */
    __shared__ volatile float shared[T];
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int l_tid = threadIdx.x;

    if(g_tid < n)
        shared[l_tid] = arr[g_tid];
    else
        shared[l_tid] = 0.;

    __syncthreads();

    _reduce_

    if(l_tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

__global__ void _sum_reduce_rows(float *mat, float *result, int r1, int c1) {
    /**
     * Sums down columns in row major matrix mat and places sums in the first 
     * row of the matrix result. r1 and c1 are the dimensions of mat.
     */
    __shared__ volatile float shared[T];
    int g_tid = blockIdx.x + c1 * threadIdx.x;
    int l_tid = threadIdx.x;

    if(l_tid < r1)
        shared[l_tid] = mat[g_tid];
    else
        shared[l_tid] = 0.;

    __syncthreads();

    _reduce_

    if(l_tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}
__global__ void _mat_mul(float *mat1, float *mat2, float *result, int c1, int c2) {
    /**
     * Multiplies row major matrices mat1 and mat2. Then it places results in result. 
     * c1 and c2 are the numbers of columns in mat1 and mat2.
     */
    __shared__ volatile float shared[T];
    int l_tid = threadIdx.x;

    if(l_tid < c1)
        shared[l_tid] = mat1[blockIdx.x / c2 * c1 + l_tid] * mat2[blockIdx.x % c2 + c2 * l_tid];
    else
        shared[l_tid] = 0;

    __syncthreads();

    _reduce_

    if(l_tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

__global__ void _sigmoid_prime(float *mat, float *result, int n) {
    /**
     * Applies the sigmoid prime function to all values in mat and places results in result. n
     * is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n) {
        float val = 0;
        val = 1. / (1. + exp(-mat[tid]));
        result[tid] = val * (1 - val);
    }
}

__global__ void _sigmoid(float *mat, float *result, int n) {
    /**
     * Applies the sigmoid function to all values in mat and places results in result. n
     * is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n) {
        float val = 0;
        val = 1. / (1. + exp(-mat[tid]));
        result[tid] = val;
    }
}

__global__ void _relu(float *mat, float *result, int n) {
    /**
     * Applies the RELU function to all values in mat and places results in result. n
     * is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n)
        if(mat[tid] <= 0.)
            result[tid] = 0.;
        else
            result[tid] = mat[tid];
}

__global__ void _elwise_subtract(float *mat1, float *mat2, float *result, int n) {
    /**
     * Subtracts each element in mat2 from the corresponding element in mat1 
     * and places results in result. n is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n)
        result[tid] = mat1[tid] - mat2[tid];
}

__global__ void _elwise_mult(float *mat1, float *mat2, float *result, int n) {
    /**
     * Multiplies each element in mat1 with the corresponding element in mat2 
     * and places results in result. n is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n)
        result[tid] = mat1[tid] * mat2[tid];
}

__global__ void _add_bias(float *mat, float *bias, int c1) {
    /**
     * Adds the 1 x c1 matrix bias to mat by broadcasting the vector across each
     * row of mat.
     */
    int mat_tid = blockIdx.x * c1 + threadIdx.x;
    int bias_tid = threadIdx.x;

    if(threadIdx.x < c1)
        mat[mat_tid] += bias[bias_tid];
}

__global__ void _update(float *mat, float *del_mat, float lr, int n) {
    /**
     * Adds to each element in mat the corresponding value in del_mat 
     * scaled by the learning rate lr. n is the number of elements in mat
     * and del_mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
        mat[tid] = mat[tid] - lr * del_mat[tid];
}


__global__ void _transpose(float *mat, float *result, int c1, int r1) {
    /**
     * Transposes the matrix mat and places the resulting matrix in result.
     * c1 is the number of columns in mat and r1 is the number of rows.
     */
    int mat_tid = (blockIdx.x * c1) + threadIdx.x;
    int result_tid = blockIdx.x + (threadIdx.x * r1);

    if(threadIdx.x < c1)
        result[result_tid] = mat[mat_tid];
}

__global__ void _divide(float *mat, float *result, float denom, int n) {
    /**
     * Divides each element of mat by denom and places each result in the 
     * matrix result. n is the number of elements in mat.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
        result[tid] = mat[tid] / denom;
}


/*Wrapper functions for the CUDA kernels that accept matrix objects.*/

void add_bias(matrix *mat, matrix *bias) {
    /** 
     * Assigns a block to each row of mat and calls the _add_bias kernel.
     * Only calls the kernel if mat and bias have the same
     * number of columns.
     */
    if(!mat->on_device || !bias->on_device) {
        printf("Make sure matrix and bias are both on device before adding them.\n");
        return;
    }
    if(mat->num_cols != bias->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
        return;
    }
    int blocks = mat->num_rows;
    int threads = T;

    _add_bias<<<blocks, threads>>>(mat->device_data, bias->device_data, mat->num_cols);
}

void update_cuda(matrix *mat, matrix *del_mat, float lr) {
    /** 
     * Calculates the mimimum number of blocks needed to perform the in-place
     * update to mat and calls the _update kernel.
     * Only calls the kernel if mat and del_mat have the same dimensions.
     */
    if(!mat->on_device || !del_mat->on_device) {
        printf("Make sure matrix and gradients are both on device before adding them.\n");
        return;
    }
    if(mat->num_rows != del_mat->num_rows || mat->num_cols != del_mat->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));

    _update<<<blocks, threads>>>(mat->device_data, del_mat->device_data, lr, n);
}

void sum_reduce(matrix *mat, matrix *result) {
    /**
     * Calculates the mimimum number of blocks needed to reduce mat
     * and then calls the _sum_reduce kernel.
     * This sum reduction will work for arrays with up to 
     * T ^ 2 elements in length.
     */
    if(!mat->on_device || !result->on_device) {
        printf("Make sure matrix is on device before summing it.\n");
        return;
    }
    int n = mat->num_vals;
    int blocks = (n % T == 0) ? (n / T) : (n / T + 1);

    _sum_reduce<<<blocks, T>>>(mat->device_data, result->device_data, n);

    n = blocks;
    
    _sum_reduce<<<1, T>>>(result->device_data, result->device_data, n);
}

void sum_reduce_rows(matrix *mat, matrix *result) {
    /** 
     * Assigns a block to each column of mat and calls the _sum_reduce_rows kernel.
     * Only calls the kernel if mat and bias have the same
     * number of columns.
     */
    if(!mat->on_device || !result->on_device) {
        printf("Make sure matrix is on device before summing it.\n");
        return;
    }
    int blocks = mat->num_cols;
    int threads = T;

    _sum_reduce_rows<<<blocks, threads>>>(mat->device_data, result->device_data, mat->num_rows, mat->num_cols);
}

void divide(matrix *mat, matrix *result, float denom) {
    /**
     * Calculates the mimimum number of blocks needed to divide each element
     * in mat by demon and then calls the _divide kernel.
     */
    if(!mat->on_device || !result->on_device) {
        printf("Make sure mat and result are on device before dividing.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));
    
    _divide<<<blocks, threads>>>(mat->device_data, result->device_data, denom, n);
}

void elwise_mult(matrix *mat1, matrix *mat2, matrix *result) {
    /**
     * Calculates the mimimum number of blocks needed to multiply each element
     * in mat1 by corresponding element in mat2 and then calls the _elwise_mult kernel.
     * Only calls kernel if mat1, mat2, and result have the same dimensions.
     */
    if(!mat1->on_device || !mat2->on_device || !result->on_device) {
        printf("Make sure mat1, mat2, result, are on device before multiplying.\n");
        return;
    }
    if(mat1->num_rows != mat2->num_rows || mat1->num_cols != mat2->num_cols || result->num_rows != mat1->num_rows || result->num_cols != mat1->num_cols) {
        printf("mat1: "); mat1->print_dims();
        printf("mat2: "); mat2->print_dims();
        printf("result: "); result->print_dims();
        return;
    }
    int n = mat1->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));
    
    _elwise_mult<<<blocks, threads>>>(mat1->device_data, mat2->device_data, result->device_data, n);
}


void elwise_subtract(matrix *mat1, matrix *mat2, matrix *result) {
    /**
     * Calculates the mimimum number of blocks needed to subtract each element
     * in mat1 by corresponding element in mat2 and then calls the _subtract kernel.
     * Only calls kernel if mat1, mat2, and result have the same dimensions.
     */
    if(!mat1->on_device || !mat2->on_device || !result->on_device) {
        printf("Make sure mat1, mat2, result, are on device before subtracting.\n");
        return;
    }
    if(mat1->num_rows != mat2->num_rows || mat1->num_cols != mat2->num_cols || result->num_rows != mat1->num_rows || result->num_cols != mat1->num_cols) {
        return;
    }
    int n = mat1->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));
    
    _elwise_subtract<<<blocks, threads>>>(mat1->device_data, mat2->device_data, result->device_data, n);
}

void mat_mul(matrix *mat1, matrix *mat2, matrix *result) {
    /** 
     * Assigns a block to an element in result and then calls the _mat_mul kernel.
     * Only calls the kernel if mat1's columns and mat2's rows are equal.
     */
    if(mat1->on_device && mat2->on_device && result->on_device) {
        int r1 = mat1->num_rows, c1 = mat1->num_cols;
        int r2 = mat2->num_rows, c2 = mat2->num_cols;
        if(c1 == r2) {
            _mat_mul<<<r1 * c2, T>>>(mat1->device_data, mat2->device_data, result->device_data, c1, c2);
        }
        else {
            printf("Incompatible matrix dimensions. m1 is %d x %d, m2 is %d x %d\n", r1, c1, r2, c2);
        }
    }
    else
        printf("Make sure input matrices and output matrix have been moved to device");
}

void activate(matrix *mat, matrix *result, int type) {
    /**
     * Calculates the mimimum number of blocks needed to activate each element
     * in mat and then calls activation kernel specified by type.
     */
    if(!mat->on_device) {
        printf("Make sure matrix is on device before activating.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));

    if(type == 0)
        _sigmoid<<<blocks, threads>>>(mat->device_data, result->device_data, n);
    else if(type == 1)
        _relu<<<blocks, threads>>>(mat->device_data, result->device_data, n);
    else if(type == 2)
        printf("Softmax has not been implemented yet.\n");
    else
        printf("This activation function has not been configured.\n");
}

void activate_prime(matrix *mat, matrix *result, int type) {
    /**
     * Calculates the mimimum number of blocks needed to "reverse" activate each element
     * in mat and then callsthe kernel for the derivative of the activation function specified by type.
     */
    if(!mat->on_device) {
        printf("Make sure matrix is on device before activating.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    int blocks = int(ceil(float(n) / threads));

    if(type == 0)
        _sigmoid_prime<<<blocks, threads>>>(mat->device_data, result->device_data, n);
    else
        printf("This activation function has not been configured.\n");
}

void transpose(matrix *mat, matrix *result) {
    /**
     * Assigns a block to each row in mat and then calls the _transpose kernel.
     * Only calls the kernel if the dimensions of mat1 are the reverse of mat2.
     */
    if(!mat->on_device || !result->on_device) {
        printf("Make sure mat, result, are on device before transposing.\n");
        return;
    }
    if(mat->num_rows != result->num_cols || mat->num_cols != result->num_rows) {
        printf("mat: "); mat->print_dims();
        printf("result: "); result->print_dims();
        return;
    }
    int blocks = mat->num_rows;
    int threads = T;
    _transpose<<<blocks, threads>>>(mat->device_data, result->device_data, mat->num_cols, mat->num_rows);
}

float MSE_mat(matrix *y, matrix *yhat, matrix *result) {
    /**
     * Calls various kernels in order to calculate the mean squared error
     * given the one-hot label vectors making up each row of y
     * and the logit outputs from the last layer of the network, yhat.
     * The mean squared error is then returned as a float. The constituent
     * wrappers and kernels are only called if y and yhat's dimensions are the same.
     */
    if(!y->on_device || !yhat->on_device || !result->on_device) {
        printf("Make sure y, yhat, result are on device before MSE.\n");
        return -1;
    }
    if(y->num_rows != yhat->num_rows || y->num_cols != yhat->num_cols) {
        printf("y: "); y->print_dims();
        printf("yhat: "); yhat->print_dims();
        printf("result: "); result->print_dims();
        return -1.;
    }
    float mse;
    elwise_subtract(y, yhat, result);
    elwise_mult(result, result, result);
    sum_reduce(result, result);
    cudaMemcpy(&mse, result->device_data, sizeof(float), cudaMemcpyDeviceToHost);
    return mse / (float(y->num_rows) * 2);
}
