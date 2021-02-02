#include <cuda.h>
#include "matrixMul.h"

__global__ void matrixMulSquareKernel(float *d_M, float *d_N,
        float *d_P, int width) {
    // P_{ij} = M_ikN_kj
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < width && col < width)) {
        int k;
        int P_ind = row * width + col;
        d_P[P_ind] = 0.0;
        for (k = 0; k < width; k++) {
            d_P[P_ind] += d_M[row*width + k]*d_N[k*width + col];
        }
    }
}

void matrixMulSquare(float *M, float *N, float *P, int width) {
    const dim3 dimBlock(16, 16);
    const int numBlocks = ceil(width/16.0);
    const dim3 dimGrid(numBlocks, numBlocks);
    float *d_M, *d_N, *d_P;

    int sz = width*width * sizeof(float);

    cudaMalloc((void **) &d_M, sz);
    cudaMalloc((void **) &d_N, sz);
    cudaMalloc((void **) &d_P, sz);

    cudaMemcpy(d_M, M, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, M, sz, cudaMemcpyHostToDevice);
    
    matrixMulSquareKernel<<<dimGrid, dimBlock>>>(d_M,
            d_N, d_P, width);

    cudaMemcpy(P, d_P, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
}
