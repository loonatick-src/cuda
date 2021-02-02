#include <cuda.h>
#include <stdio.h>
#include "matrixAdd.h"
#include "matrixMul.h"
#include "matrixIO.h"


int main() {
    int n;
    scanf("%d", &n);
    int sz = n*n;
    float *A, *B, *D;
    A = (float *) malloc(sz*sizeof(float));
    B = (float *) malloc(sz*sizeof(float));

    readMat(A,n);
    readMat(B,n);
#ifdef TESTIO
    printMat(A, n);
    printMat(B, n);
#endif

#ifdef ADD
    float* C = (float *) malloc(sz*sizeof(float));
    matrixAdd(A, B, C, n);
    printMat(C, n);
#endif

#ifndef TESTIO
    D = (float *) malloc(sz*sizeof(float));
    matrixMulSquare(A, B, D, n);
    printMat(D, n);
#endif

    return 0;
}
