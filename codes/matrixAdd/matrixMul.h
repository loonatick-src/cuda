#ifndef MATRIXMUL_H
#define MATRIXMUL_H
__global__ void matrixMulSquareKernel(float *d_M, float *d_N,
        float* d_P, int width);

void matrixMulSquare(float *M, float *N, float *P, int width);
#endif
