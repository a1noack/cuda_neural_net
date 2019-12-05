#include <stdio.h>
#include "matrix.hpp"

// make sure T is bigger than the number of neurons in the largest layer
#define T 128

/*Cuda kernels*/

__global__ void _sum_reduce1(float *arr, int n) {
    __shared__ float shared[T];
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int l_tid = threadIdx.x;

    if(g_tid < n)
        shared[l_tid] = arr[g_tid];
    else
        shared[l_tid] = 0.;

    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2) {
        if(l_tid % (2 * s) == 0) {
            shared[l_tid] = shared[l_tid] + shared[l_tid + s];
        }

        __syncthreads();
    }

    if(l_tid == 0) {
        arr[blockIdx.x] = shared[0];
    }
}

__global__ void _mat_mul(float *mat1, float *mat2, float *result, int c1, int c2) {
    __shared__ float shared[T];
    int tid = threadIdx.x;
     
    if(tid < c1)
        shared[tid] = mat1[blockIdx.x / c2 * c1 + tid] * mat2[blockIdx.x % c2 + c2 * tid];
    else
        shared[tid] = 0;        
    
    __syncthreads();

    for(int s = 1; s < blockDim.x; s*=2) {
        if(tid % (2 * s) == 0) {
            shared[tid] = shared[tid] + shared[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

__global__ void _sigmoid_prime(float *mat, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        float val = 0;
        val = 1. / (1. + exp(-mat[tid]));
        result[tid] = val * (1 - val);
    }
}

__global__ void _sigmoid(float *mat, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        float val = 0;
        val = 1. / (1. + exp(-mat[tid]));
        result[tid] = val;
    }
}

__global__ void _relu(float *mat, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        if(mat[tid] <= 0.)
            result[tid] = 0.;
        else
            result[tid] = mat[tid];
}

__global__ void _elwise_subtract(float *mat1, float *mat2, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = mat1[tid] - mat2[tid];
}

__global__ void _elwise_mult(float *mat1, float *mat2, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = mat1[tid] * mat2[tid];
}

__global__ void _add_bias(float *mat, float *bias, int c1) {
    int mat_tid = blockIdx.x * c1 + threadIdx.x;
    int bias_tid = threadIdx.x;
    if(threadIdx.x < c1)
        mat[mat_tid] += bias[bias_tid];
}

__global__ void _update(float *mat, float *del_mat, float lr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        mat[tid] = mat[tid] - lr * del_mat[tid];
}


__global__ void _transpose(float *mat, float *result, int c1, int r1) {
    int mat_tid = (blockIdx.x * c1) + threadIdx.x;
    int result_tid = blockIdx.x + (threadIdx.x * r1);
    if(threadIdx.x < c1)
        result[result_tid] = mat[mat_tid];
}

__global__ void _divide(float *mat, float *result, float denom, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = mat[tid] / denom;
}

/*Wrapper functions for the CUDA kernels that accept matrix objects.*/

void add_bias(matrix *mat, matrix *bias) {
    if(!mat->on_device || !bias->on_device) { 
        printf("Make sure matrix and bias are both on device before adding them.\n");
        return;
    }
    if(mat->num_cols != bias->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
    }
    bias->print_dims();
    int blocks = mat->num_rows;
    int threads = T;
    _add_bias<<<blocks, threads>>>(mat->device_data, bias->device_data, mat->num_cols);
}


void update(matrix *mat, matrix *del_mat, float lr) {
    if(!mat->on_device || !del_mat->on_device) { 
        printf("Make sure matrix and gradients are both on device before adding them.\n");
        return;
    }
    if(mat->num_rows != del_mat->num_rows || mat->num_cols != del_mat->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
    }
    int n = mat->num_vals;
    int threads = T;
    //int blocks = mat->num_rows;
    int blocks = int(ceil(float(n) / threads));
    //int blocks = mat->num_rows;
    //int threads = T;
    _update<<<blocks, threads>>>(mat->device_data, del_mat->device_data, lr, n);
}

float *sum_reduce(matrix *mat) {
    if(!mat->on_device) { 
        printf("Make sure matrix and gradients are both on device before adding them.\n");
        return NULL;
    }
    int n = mat->num_vals;
    /*This sum reduction will work for arrays with up to 
     T ^ 2 = 128 * 128 = 16384 elements in length.*/
    int blocks = (n % T == 0) ? (n / T) : (n / T + 1);
    
    _sum_reduce1<<<blocks, T>>>(mat->device_data, n);

    n = blocks;
    _sum_reduce1<<<1, T>>>(mat->device_data, n);
    return mat->device_data;
}

void divide(matrix *mat, matrix *result, float denom) {
    if(!mat->on_device || !result->on_device) {
        printf("Make sure mat and result are on device before dividing.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    //int blocks = mat->num_rows;
    int blocks = int(ceil(float(n) / threads));
    //int blocks = mat->num_rows;
    //int threads = T;
    _divide<<<blocks, threads>>>(mat->device_data, result->device_data, denom, n);
}

void elwise_mult(matrix *mat1, matrix *mat2, matrix *result) {
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
    //int blocks = mat->num_rows;
    int blocks = int(ceil(float(n) / threads));
    //int blocks = mat1->num_rows;
    //int threads = T;
    _elwise_mult<<<blocks, threads>>>(mat1->device_data, mat2->device_data, result->device_data, n);
}


void elwise_subtract(matrix *mat1, matrix *mat2, matrix *result) {
    if(!mat1->on_device || !mat2->on_device || !result->on_device) { 
        printf("Make sure mat1, mat2, result, are on device before subtracting.\n");
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
    //int blocks = mat->num_rows;
    int blocks = int(ceil(float(n) / threads));
    //int blocks = mat1->num_rows;
    //int threads = T;
    _elwise_subtract<<<blocks, threads>>>(mat1->device_data, mat2->device_data, result->device_data, n);
}

void mat_mul(matrix *mat1, matrix *mat2, matrix *result) {
    if(mat1->on_device && mat2->on_device && result->on_device) {
        int r1 = mat1->num_rows, c1 = mat1->num_cols;
        int r2 = mat2->num_rows, c2 = mat2->num_cols;
        if(c1 == r2)
            _mat_mul<<<r1 * c2, T>>>(mat1->device_data, mat2->device_data, result->device_data, c1, c2);
        else
            printf("Incompatible matrix dimensions. m1 is %d x %d, m2 is %d x %d\n", r1, c1, r2, c2);
    }
    else
        printf("Make sure input matrices and output matrix have been moved to device");
}

void activate(matrix *mat, matrix *result, int type) {
    if(!mat->on_device) { 
        printf("Make sure matrix is on device before activating.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    //int blocks = mat->num_rows;
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
    if(!mat->on_device) { 
        printf("Make sure matrix is on device before activating.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = T;
    //int blocks = mat->num_rows;
    int blocks = int(ceil(float(n) / threads));

    if(type == 0)
        _sigmoid_prime<<<blocks, threads>>>(mat->device_data, result->device_data, n);
    else
        printf("This activation function has not been configured.\n");
} 

void transpose(matrix *mat, matrix *result) {
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
