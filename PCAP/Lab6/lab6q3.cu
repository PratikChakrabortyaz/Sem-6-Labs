#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 16  

__global__ void odd_even(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % 2 != 0 && tid < n - 1) {
        if (arr[tid] > arr[tid + 1]) {
            int temp = arr[tid];
            arr[tid] = arr[tid + 1];
            arr[tid + 1] = temp;
        }
    }
}

__global__ void even_odd(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % 2 == 0 && tid < n - 1) {
        if (arr[tid] > arr[tid + 1]) { 
            int temp = arr[tid];
            arr[tid] = arr[tid + 1];
            arr[tid + 1] = temp;
        }
    }
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
    
    int *h_arr, *d_arr;
    size_t bytes = n * sizeof(int);
    
    h_arr = (int*)malloc(bytes);
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_arr[i]);
    }
    
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    
    dim3 dimGrid((int)ceil((float)n/BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE);

    for (int i = 0; i < n / 2; i++) {
        odd_even<<<dimGrid, dimBlock>>>(d_arr, n);
        cudaDeviceSynchronize();
        even_odd<<<dimGrid, dimBlock>>>(d_arr, n);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    
    printf("Sorted array: \n");
    printArray(h_arr, n);
    
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
