#include <stdio.h>
#include <cuda.h>

__global__ void sparseMatrixVectorMul(int *row_ptr, int *col_ind, float *val, float *vec, float *result, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
            sum += val[i] * vec[col_ind[i]];
        }
        result[row] = sum;
    }
}

int main() {
    int h_row_ptr[] = {0, 2, 4, 5};
    int h_col_ind[] = {0, 1, 0, 2, 1};
    float h_val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_vec[] = {1.0f, 2.0f, 3.0f};
    int rows = 3;

    int *d_row_ptr, *d_col_ind;
    float *d_val, *d_vec, *d_result;
    cudaMalloc((void**)&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_ind, 5 * sizeof(int));
    cudaMalloc((void**)&d_val, 5 * sizeof(float));
    cudaMalloc((void**)&d_vec, 3 * sizeof(float));
    cudaMalloc((void**)&d_result, rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, 3 * sizeof(float), cudaMemcpyHostToDevice);

    sparseMatrixVectorMul<<<(rows + 255) / 256, 256>>>(d_row_ptr, d_col_ind, d_val, d_vec, d_result, rows);

    float h_result[rows];
    cudaMemcpy(h_result, d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector:\n");
    for (int i = 0; i < rows; i++) {
        printf("%f\n", h_result[i]);
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_val);
    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}
