#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 16  

__global__ void selectionSortKernel(int *arr, int *O, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int data = arr[tid];
    int pos = 0;

    for (int i = 0; i < n; i++) {
        if ((arr[i] < data) || (arr[i] == data && i < tid)) {
            pos++;
        }
    }
    O[pos] = data;
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);
    
    int *h_arr, *h_output, *d_arr, *d_output;
    size_t bytes = n * sizeof(int);
    
    h_arr = (int*)malloc(bytes);
    h_output = (int*)malloc(bytes);
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_arr[i]);
    }
    
    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    
    dim3 dimGrid((int)ceil((float)n / BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE);
    
    selectionSortKernel<<<dimGrid, dimBlock>>>(d_arr, d_output, n);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    printf("Sorted array: \n");
    printArray(h_output, n);
    
    cudaFree(d_arr);
    cudaFree(d_output);
    free(h_arr);
    free(h_output);
    return 0;
}
