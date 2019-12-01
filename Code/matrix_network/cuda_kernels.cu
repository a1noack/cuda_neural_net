#include <stdio.h>
#include "matrix.hpp"

#define T 128

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

__global__ void _sigmoid(float *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        mat[tid] = 1. / (1. + exp(-mat[tid]));
}

__global__ void _relu(float *mat, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n && mat[tid] <= 0)
        mat[tid] = 0;
}

void activate(matrix *mat, int type) {
    if(!mat->on_device) { 
        printf("Make sure matrix is on device before activating.\n");
        return;
    }
    int n = mat->num_vals;
    int threads = 128;
    int blocks = int(ceil(float(n) / threads));
    if(type == 0)
        _sigmoid<<<blocks, threads>>>(mat->device_data, n);
    else if(type == 1)
        _relu<<<blocks, threads>>>(mat->device_data, n);
    else if(type == 2)
        printf("Softmax has not been implemented yet.\n");
    else
        printf("This activation function has not been configured.\n");
} 

__global__ void _add_bias(float *mat, float *bias, int num_cols) {
    int mat_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bias_tid = threadIdx.x;
    if(bias_tid < num_cols)
        mat[mat_tid] += bias[bias_tid];
}

void add_bias(matrix *mat, matrix *bias) {
    if(!mat->on_device || !bias->on_device) { 
        printf("Make sure matrix and bias are both on device before adding them.\n");
        return;
    }
    if(mat->num_cols != bias->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
    }
    int blocks = mat->num_rows;
    int threads = T;
    _add_bias<<<blocks, threads>>>(mat->device_data, bias->device_data, mat->num_cols);
}

__global__ void _update(float *mat, float *del_mat, float lr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        mat[tid] = mat[tid] - lr * del_mat[tid];
}

void update(matrix *mat, matrix *del_mat, float lr) {
    if(!mat->on_device || !del_mat->on_device) { 
        printf("Make sure matrix and gradients are both on device before adding them.\n");
        return;
    }
    if(mat->num_rows != del_mat->num_rows || mat->num_cols != del_mat->num_cols) {
        printf("mat and del_mat don't have the same dimensions.\n");
    }
    int blocks = mat->num_rows;
    int threads = T;
    _update<<<blocks, threads>>>(mat->device_data, del_mat->device_data, lr, mat->num_vals);
}

__global__ void sum_reduce1(float *arr, int n) {
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

void sum_reduce(float *arr, int n) {
    /*This sum reduction will work for arrays with up to 
     T ^ 2 = 128 * 128 = 16384 elements in length.*/
    int blocks = (n % T == 0) ? (n / T) : (n / T + 1);
    
    sum_reduce1<<<blocks, T>>>(arr, n);

    n = blocks;
    sum_reduce1<<<1, T>>>(arr, n);
}

__global__ void elwise_mult(float **arr1, float **arr2, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = *arr1[tid] * *arr2[tid];
}
