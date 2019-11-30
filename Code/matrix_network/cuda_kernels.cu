#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include "matrix.hpp"
#include <math.h>

#define T 128

void print_array(float *a, int n) {
    for(int i = 0; i < n; i++) {
        printf("%.1f ", a[i]);
    }
    printf("\n");
}

void print_dev_array(float *a, int n) {
    float temp[n];
    cudaMemcpy(temp, a, n * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(temp, n);
}

void populate_array(float *a, int n) {
    for(int i = 0; i < n; i++)
        a[i] = rand() % 10;    
}

__global__ void elwise_mult(float *arr1, float *arr2, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    result[tid] = arr1[tid] * arr2[tid];
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

float dot(float *arr1, float *arr2, float *result, int n) { 
    float d;
    int blocks = (n % T == 0) ? (n / T) : (n / T + 1);
    elwise_mult<<<blocks, T>>>(arr1, arr2, result, n);
    sum_reduce(result, n);
    cudaMemcpy(&d, result, sizeof(float), cudaMemcpyDeviceToHost);
    return d;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    if(argc != 2) {
        printf("enter n as an argument\n");
        exit(0);
    }
    int n = atoi(argv[1]);

    float *a = new float[n];
    float *b = new float[n];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    populate_array(a, n);
    populate_array(b, n);
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
   
    float dp = dot(d_a, d_b, d_c, n);
    printf("a = "); print_dev_array(d_a, n);
    printf("b = "); print_dev_array(d_b, n);
    printf("dot product = %f\n", dp);
    exit(0);
}
