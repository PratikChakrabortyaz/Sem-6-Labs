#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define TILE_WIDTH 16
#define MAX_MASK_WIDTH 64  

__constant__ int d_mask[MAX_MASK_WIDTH]; 

__global__ void convolution1D_tiled(int *input, int *output, int width, int mask_width) {
    extern __shared__ int N_s[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    int r = mask_width / 2;  
    int shared_idx = tid + r;

    // Load main element
    if (i < width)
        N_s[shared_idx] = input[i];
    else
        N_s[shared_idx] = 0;

    // Load left halo
    if (tid < r) {
        int halo_idx = i - r;
        N_s[tid] = (halo_idx >= 0) ? input[halo_idx] : 0;
    }

    // Load right halo
    if (tid >= blockDim.x - r) {
        int halo_idx = i + r;
        if (halo_idx < width)
            N_s[shared_idx + r] = input[halo_idx];
        else
            N_s[shared_idx + r] = 0;
    }

    __syncthreads();

    // Compute convolution
    int sum = 0;
    if (i < width) {
        for (int j = 0; j < mask_width; j++) {
            sum += d_mask[j] * N_s[shared_idx - r + j];
        }
        output[i] = sum;
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int width, mask_width;

    printf("Enter the size of the input array: ");
    scanf("%d", &width);
    printf("Enter the size of the mask array (<= %d): ", MAX_MASK_WIDTH);
    scanf("%d", &mask_width);

    if (mask_width > MAX_MASK_WIDTH) {
        printf("Error: mask size exceeds maximum allowed size for constant memory.\n");
        return 1;
    }

    int *h_input, *h_mask, *h_output;
    int *d_input, *d_output;

    size_t bytes_input = width * sizeof(int);
    size_t bytes_mask = mask_width * sizeof(int);

    h_input = (int *)malloc(bytes_input);
    h_mask = (int *)malloc(bytes_mask);
    h_output = (int *)malloc(bytes_input);

    printf("Enter the elements of the input array: ");
    for (int i = 0; i < width; i++) {
        scanf("%d", &h_input[i]);
    }

    printf("Enter the elements of the mask array: ");
    for (int i = 0; i < mask_width; i++) {
        scanf("%d", &h_mask[i]);
    }

    cudaMalloc(&d_input, bytes_input);
    cudaMalloc(&d_output, bytes_input);

    cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, bytes_mask);

    int shared_mem_size = (TILE_WIDTH + mask_width - 1) * sizeof(int);

    dim3 dimGrid((int)ceil((float)width / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH);

    convolution1D_tiled<<<dimGrid, dimBlock, shared_mem_size>>>(d_input, d_output, width, mask_width);

    cudaMemcpy(h_output, d_output, bytes_input, cudaMemcpyDeviceToHost);

    printf("Output Array:\n");
    printArray(h_output, width);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}
