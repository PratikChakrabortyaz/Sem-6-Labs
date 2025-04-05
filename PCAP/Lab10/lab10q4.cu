#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__global__ void inclusive_scan(int *d_in, int *d_out, int n) {
    __shared__ int temp[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) 
    	temp[threadIdx.x] = d_in[tid];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (threadIdx.x >= offset) val = temp[threadIdx.x - offset];
        __syncthreads();
        if (threadIdx.x >= offset) temp[threadIdx.x] += val;
        __syncthreads();
    }
    
    if (tid < n) d_out[tid] = temp[threadIdx.x];
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);
    int *h_in = (int*)malloc(n * sizeof(int));
    int *h_out = (int*)malloc(n * sizeof(int));

    printf("Enter %d elements: ", n);
    for (int i = 0; i < n; i++) scanf("%d", &h_in[i]);

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    inclusive_scan<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Inclusive scan result: ");
    for (int i = 0; i < n; i++) printf("%d ", h_out[i]);
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}