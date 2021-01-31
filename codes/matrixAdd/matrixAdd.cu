#include <cuda.h>
#include "matrixAdd.h"

__global__ 
void matrixAddKernel_elementwise(
        float *A, float *B, float *C, int n) {
    // A, B, and C are device memory pointers
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i < n && j < n) {
        int k = n*i + j;
        C[k] = A[k] + B[k];
    }
}


void matrixAdd(float *A, float *B, 
        float *C, int n) {
    int size = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;  // device memory pointers
    const dim3 threadsPerBlock(16, 16);  // 256 threads per block
    const dim3 numBlocks(ceil(n/threadsPerBlock.x), ceil(n/threadsPerBlock.y));

    cudaMalloc((void**) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size);

    matrixAddKernel_elementwise<<<numBlocks, threadsPerBlock>>>(
            d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);
}
