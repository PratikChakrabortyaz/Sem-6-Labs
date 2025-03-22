#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MAX_SIZE 1024  

__global__ void multRowWise(int *A, int *B, int *C, int ha, int wa, int wb) {
    int ridA = threadIdx.x;  
    int cidB = threadIdx.y;  
    int sum = 0;

    if (ridA < ha && cidB < wb) {
        for (int k = 0; k < wa; k++) {
            sum += A[ridA * wa + k] * B[k * wb + cidB];
        }
        C[ridA * wb + cidB] = sum;
    }
}

__global__ void multColumnWise(int *A, int *B, int *C, int ha, int wa, int wb) {
    int cidB = threadIdx.x; 
    int ridA = threadIdx.y; 
    int sum = 0;

    if (ridA < ha && cidB < wb) {
        for (int k = 0; k < wa; k++) {
            sum += A[ridA * wa + k] * B[k * wb + cidB];
        }
        C[ridA * wb + cidB] = sum;
    }
}

__global__ void multElementWise(int *A, int *B, int *C, int ha, int wa, int wb) {
    int ridA = blockIdx.x * blockDim.x + threadIdx.x; 
    int cidB = blockIdx.y * blockDim.y + threadIdx.y;  
    int sum = 0;

    if (ridA < ha && cidB < wb) {
        for (int k = 0; k < wa; k++) {
            sum += A[ridA * wa + k] * B[k * wb + cidB];
        }
        C[ridA * wb + cidB] = sum;
    }
}

void inputMatrix(int *matrix, int height, int width) {
    printf("Enter the elements of the matrix (%dx%d):\n", height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("Element [%d][%d]: ", i, j);
            scanf("%d", &matrix[i * width + j]);
        }
    }
}

void printMatrix(int *matrix, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C_row, *C_col, *C_elem;
    int *d_A, *d_B, *d_C_row, *d_C_col, *d_C_elem;
    int ha, wa, hb, wb;

    printf("Enter dimensions of Matrix A (rows cols):\n");
    scanf("%d %d", &ha, &wa);
    printf("Enter dimensions of Matrix B (rows cols):\n");
    scanf("%d %d", &hb, &wb);

    if (wa != hb) {
        printf("Matrix multiplication not possible. Columns of A must match rows of B.\n");
        return 1;
    }

    A = (int *)malloc(ha * wa * sizeof(int));
    B = (int *)malloc(hb * wb * sizeof(int));
    C_row = (int *)malloc(ha * wb * sizeof(int));
    C_col = (int *)malloc(ha * wb * sizeof(int));
    C_elem = (int *)malloc(ha * wb * sizeof(int));

    printf("Matrix A:\n");
    inputMatrix(A, ha, wa);
    printf("Matrix B:\n");
    inputMatrix(B, hb, wb);

    cudaMalloc(&d_A, ha * wa * sizeof(int));
    cudaMalloc(&d_B, hb * wb * sizeof(int));
    cudaMalloc(&d_C_row, ha * wb * sizeof(int));
    cudaMalloc(&d_C_col, ha * wb * sizeof(int));
    cudaMalloc(&d_C_elem, ha * wb * sizeof(int));

    cudaMemcpy(d_A, A, ha * wa * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, hb * wb * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock1(ha, wb, 1);
    multRowWise<<<1, dimBlock1>>>(d_A, d_B, d_C_row, ha, wa, wb);

    dim3 dimBlock2(ha, wb, 1);
    multColumnWise<<<1, dimBlock2>>>(d_A, d_B, d_C_col, ha, wa, wb);

    dim3 dimGrid(ceil(ha/16.0),ceil(wb/16.0),1);
    dim3 dimBlock3(16,16,1);
    multElementWise<<<dimGrid,dimBlock3>>>(d_A, d_B, d_C_elem, ha, wa, wb);

    cudaMemcpy(C_row, d_C_row, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_col, d_C_col, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_elem, d_C_elem, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix (Row-wise multiplication):\n");
    printMatrix(C_row, ha, wb);
    printf("Resultant Matrix (Column-wise multiplication):\n");
    printMatrix(C_col, ha, wb);
    printf("Resultant Matrix (Element-wise multiplication):\n");
    printMatrix(C_elem, ha, wb);

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
