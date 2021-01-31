#ifndef VECADD_H
#define VECADD_H
__global__
void vecAddKernel(float *A, float *B, float *C, int n);

void vecAdd(float *A, float *B, float *C, int n);

#endif
