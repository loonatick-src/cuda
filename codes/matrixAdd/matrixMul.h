#ifndef MATRIXMUL_H
#define MATRIXMUL_H
__global__
void
matrixMulSquareKernel(float *d_M, float *d_N,
        float* d_P, int width);

void
matrixMulSquare(float *M, float *N, float *P, int width);

__global__
void
matrix_vectorMulKernel(float *A, float *v, float *p, int n);

void
matrix_vectorMul(float *A, float *v, float *P, int n);

#endif
