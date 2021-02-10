#include <cuda.h>
#include "matrixAdd.h"

// TODO: remove boilerplate using preprocessor macros
__global__ 
void
matrixAddKernel_elementwise(
        float *A, float *B, float *C, int n) {
    // A, B, and C are device memory pointers
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i < n && j < n) {
        int k = n*i + j;
        C[k] = A[k] + B[k];
    }
}


void
matrixAdd(float *A, float *B, 
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


__global__
void
matrixAddKernel_roww(float *A, float *B, float *C, int n) {
    // each thread performs matrix addition for one row
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    if (row < n) {
        for (int i = 0; i < n; i++) {
            int ind = n*row + i;
            C[ind] = A[ind] + B[ind];
        }
    }
}


void
matrixAdd_roww(float *A, float *B, float *C, int n) {
    float *d_A, *d_B, *d_C;
    const int threadsPerBlock = 128;
    const int numBlocks = (int) ceil(((float)n)/threadsPerBlock);
    const int size = n*n;

    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    matrixAddKernel_roww<<<numBlocks, threadsPerBlock>>>(
            d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);
}


__global__
void
matrixAddKernel_colw(float *A, float *B, float *C, int n) {
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    if (col < n) {
        for (int i = 0; i < n; i ++) {
            int index = n*i + col;
            C[index] = A[index] + B[index];
        }
    }
}


void
matrixAdd_colw(float *A, float *B, float *C, int n) {
    float *d_A, *d_B, *d_C;
    const int threadsPerBlock = 128;
    const int numBlocks = (int) ceil(((float)n)/threadsPerBlock);
    const int size = n*n;

    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    matrixAddKernel_colw<<<numBlocks, threadsPerBlock>>>(
            d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B), cudaFree(d_C);
}
