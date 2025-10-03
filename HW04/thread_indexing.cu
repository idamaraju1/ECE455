//Task: Launch a kernel with a 2D grid and block. Each thread should print its (blockIdx.x, blockIdx.y)
//and (threadIdx.x, threadIdx.y)

#include <stdio.h>

__global__ void print_indices() {
    printf("BlockIdx: (%d, %d), ThreadIdx: (%d, %d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main() {
    dim3 grid(2, 2);
    dim3 block(4, 4);
    print_indices<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}