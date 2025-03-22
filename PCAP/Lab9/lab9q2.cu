#include <stdio.h>
#include <cuda.h>

__global__ void transformRows(int *matrix, int rows, int cols) {
    int row = blockIdx.x; 

    if (row < rows) {
        int power = row + 1;

        for (int col = 0; col < cols; col++) {
            int value = matrix[row * cols + col];

            int result = 1;
            for (int i = 0; i < power; i++) {
                result *= value;
            }
            
            matrix[row * cols + col] = result;
        }
    }
}

int main() {
    int rows = 4; 
    int cols = 4; 

    int h_matrix[rows][cols] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 2},
        {9, 1, 2, 5}
    };

    printf("Input Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_matrix[i][j]);
        }
        printf("\n");
    }

    int *d_matrix;
    cudaMalloc((void**)&d_matrix, rows * cols * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(int), cudaMemcpyHostToDevice);


    dim3 dimGrid(rows);    
    dim3 dimBlock(1);      

    // Launch kernel
    transformRows<<<dimGrid, dimBlock>>>(d_matrix, rows, cols);

    cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_matrix[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    return 0;
}