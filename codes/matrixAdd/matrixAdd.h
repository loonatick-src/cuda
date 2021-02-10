#ifndef MATRIXADD_H
#define MATRIXADD_H

__global__
void
martixAddKernel_elementwise(float *A, float *B, float *C, int n);

void
matrixAdd(float *A, float *B, float *C, int n);

__global__
void
matrixAddKernel_roww(float *A, float *B, float *C, int n);

void
matrixAdd_roww(float *A, float *B, float *C, int n);

__global__
void
matrixAddKernel_colw(float *A, float *B, float *C, int n);

void
matrixAdd_colw(float *A, float *B, float *C, int n);

#endif
