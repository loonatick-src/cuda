#ifndef MATRIXADD_H
#define MATRIXADD_H

__global__
void martixAddKernel(float *A, float *B, float *C, int n);

void matrixAdd(float *A, float *B, float *C, int n);
#endif
