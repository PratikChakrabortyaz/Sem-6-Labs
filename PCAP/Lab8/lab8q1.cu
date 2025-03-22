#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MAX_SIZE 1024  

#define N 4  

__global__ void addRowWise(int *A, int *B, int *C, int width) {
    int row = blockIdx.x; 
    int col = threadIdx.x; 

    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void addColumnWise(int *A, int *B, int *C, int width) {
    int col = blockIdx.x;  
    int row = threadIdx.x; 

    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void addElementWise(int *A, int *B, int *C, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

void inputMatrix(int *matrix, int width) {
    printf("Enter the elements of the matrix (%dx%d):\n", width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("Element A[%d][%d]: ", i, j);
            scanf("%d", &matrix[i * width + j]);
        }
    }
}

void printMatrix(int *matrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C_row, *C_col, *C_elem;
    int *d_A, *d_B, *d_C_row, *d_C_col, *d_C_elem;

    A = (int *)malloc(N * N * sizeof(int));
    B = (int *)malloc(N * N * sizeof(int));
    C_row = (int *)malloc(N * N * sizeof(int));
    C_col = (int *)malloc(N * N * sizeof(int));
    C_elem = (int *)malloc(N * N * sizeof(int));

    printf("Matrix A:\n");
    inputMatrix(A, N);
    printf("Matrix B:\n");
    inputMatrix(B, N);

    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C_row, N * N * sizeof(int));
    cudaMalloc(&d_C_col, N * N * sizeof(int));
    cudaMalloc(&d_C_elem, N * N * sizeof(int));

    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    addRowWise<<<N, N>>>(d_A, d_B, d_C_row, N);

    addColumnWise<<<N, N>>>(d_A, d_B, d_C_col, N);

    dim3 dimGrid(ceil(N/16.0),ceil(N/16.0),1);
    dim3 dimBlock(16,16,1);
    addElementWise<<<dimGrid,dimBlock>>>(d_A, d_B, d_C_elem, N);

    cudaMemcpy(C_row, d_C_row, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_col, d_C_col, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_elem, d_C_elem, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix (Row-wise addition):\n");
    printMatrix(C_row, N);
    printf("Resultant Matrix (Column-wise addition):\n");
    printMatrix(C_col, N);
    printf("Resultant Matrix (Element-wise addition):\n");
    printMatrix(C_elem, N);

    free(A);
    free(B);
    free(C_row);
    free(C_col);
    free(C_elem);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_row);
    cudaFree(d_C_col);
    cudaFree(d_C_elem);

    return 0;
}
