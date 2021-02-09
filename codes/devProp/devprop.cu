#include <cuda.h>
#include <stdio.h>

int main(int argc, char **argv) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop,0);
    printf("Max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Multiprocessor count: %d\n", dev_prop.multiProcessorCount);
    printf("Max threads dim %d\n", dev_prop.maxThreadsDim[0]);
    printf("Max grid size (x): %d\n", dev_prop.maxGridSize[0]);
    printf("Max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", dev_prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", dev_prop.warpSize);
    return 0;
}
