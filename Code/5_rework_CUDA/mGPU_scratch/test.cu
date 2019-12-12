#include <stdio.h>
#include <cuda_runtime.h>
const int MAX_GPU_COUNT = 2;

int main() {
    printf("Checking Num GPUs\n");

    int GPU_N;
    cudaGetDeviceCount(&GPU_N);

    printf("CUDA-Capable device count: %d\n", GPU_N);
}

