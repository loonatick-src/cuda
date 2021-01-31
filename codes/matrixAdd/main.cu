#include <cuda.h>
#include <stdio.h>
#include "matrixAdd.h"
#include "matrixIO.h"


int main() {
    int n;
    scanf("%d", &n);
    int sz = n*n;
    float *A, *B, *C;
    A = (float *) malloc(sz*sizeof(float));
    B = (float *) malloc(sz*sizeof(float));
    C = (float *) malloc(sz*sizeof(float));

    readMat(A,n);
    readMat(B,n);

    matrixAdd(A, B, C, n);
    printMat(C, n);

    return 0;
}
