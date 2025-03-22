#include <stdio.h>
#include <cuda.h>

#define M 4
#define N 4

__device__ int calculateBinaryComplement(int value) {
    int binaryComplement = 0;
    int place = 1;

    while (value > 0) {
        int bit = value % 2;
        binaryComplement += (1 - bit) * place;
        place *= 10;
        value /= 2;
    }
    return binaryComplement;
}

__global__ void complement_kernel(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
            int value = A[row * cols + col];
            int comp = calculateBinaryComplement(value);
            B[row * cols + col] = comp;
        } else {
            B[row * cols + col] = A[row * cols + col];
        }
    }
}

int main() {
    int A[M][N] = {{1, 2, 3, 4}, {6, 5, 8, 3}, {2, 4, 10, 1}, {9, 1, 2, 5}};
    int B[M][N] = {0};
    int *d_A, *d_B;

    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(2, 2);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    complement_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Output matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}