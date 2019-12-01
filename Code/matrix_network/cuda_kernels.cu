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

__global__ void elwise_mult(float **arr1, float **arr2, float *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = *arr1[tid] * *arr2[tid];
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

float dot(float **arr1, float **arr2, float *result, int n) { 
    float d;
    int blocks = (n % T == 0) ? (n / T) : (n / T + 1);
    elwise_mult<<<blocks, T>>>(arr1, arr2, result, n);
    print_dev_array(result, n);
    sum_reduce(result, n);
    cudaMemcpy(&d, result, sizeof(float), cudaMemcpyDeviceToHost);
    return d;
}

void dep_mat_mul(matrix *mat1, matrix *mat2) {
    float **row, **col;
    float *res;
    int n = mat1->num_cols; // or, equivalently, mat2->num_rows 
    cudaMalloc(&res, n * sizeof(float));
    for(int r = 0; r < mat1->num_rows; r++) {
        for(int c = 0; c < mat2->num_cols; c++) {
            //printf("here\n");
            row = mat1->get_row(r);
            col = mat2->get_col(c);
            float dp = dot(row, col, res, n);
            //printf("%.1f ", dp); 
        }
        printf("\n");
    }
}

__global__ void _mat_mul(float *mat1, float *mat2, float*result, int c1, int c2) {
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
        printf("make sure input matrices and output matrix have been moved to device");
}


__global__ void test2(float **arr1, float *result, int n) {
    int tid = blockIdx.x * gridDim.x + threadIdx.x;
    if(tid < n)
        result[tid] = *arr1[tid];    
}
// dynamic parallelism needs different compilation flags and settings
//__global__ void test1(float *arr1, float *arr2, float *result, int n) {
//    elwise_mult<<<4, 32>>>(arr1, arr2, result, n);
//}

int main(int argc, char *argv[]) {
    /*int n = 4;
    float d[] = {5, 4, 3, 2};
    float *dd;
    cudaMalloc(&dd, n * sizeof(float));
    cudaMemcpy(dd, d, n * sizeof(float), cudaMemcpyHostToDevice);

    float *d2[] = {&dd[0], &dd[1], &dd[2], &dd[3]};
    float **dd2;
    cudaMalloc(&dd2, n * sizeof(float*));
    cudaMemcpy(dd2, d2, n * sizeof(float*), cudaMemcpyHostToDevice);

    float *res;
    cudaMalloc(&res, n * sizeof(float));
    printf("here\n");
    test2<<<1, 32>>>(dd2, res, n);
    print_dev_array(res, n);*/
    int r1 = 1, c1 = 15, r2 = 15, c2 = 4;

    float m1dat[r1*c1] = {0};
    float m2dat[r2*c2] = {0};

    for(int i = 0; i < r1*c1; i++) {
        m1dat[i] = 4.;
    }

    for(int i = 0; i < r2*c2; i++) {
        m2dat[i] = 3.;
    }

    matrix *m1 = new matrix(r1, c1);
    matrix *m2 = new matrix(r2, c2);
    matrix *m3 = new matrix(r1, c2);

    m1->set_memory(m1dat);
    m2->set_memory(m2dat);
    m3->set_mem_zero();

    m1->move_to_device();
    m2->move_to_device();
    m3->move_to_device();

    m1->print();
    m2->print();
    m3->print();

    mat_mul(m1, m2, m3);
    
    m3->print();

    
    /*srand(time(NULL));
    if(argc != 2) {
        printf("enter n as an argument\n");
        exit(0);
    }
    int n = atoi(argv[1]);*/


    /*float *a = new float[n];
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
    printf("c = "); print_dev_array(d_c, n);
    printf("dot product = %f\n", dp);*/
    exit(0);
}
