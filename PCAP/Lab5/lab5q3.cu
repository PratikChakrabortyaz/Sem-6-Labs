#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void computeSine(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sin(input[idx]);
    }
}

int main() {
    int N = 1024; 
    float *input, *output;
    float *d_input, *d_output;


    input = new float[N];
    output = new float[N];


    for (int i = 0; i < N; i++) {
        input[i] = (float)i * 0.01f; 
    }


    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));


    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(256, 1, 1); 


    dim3 dimGrid(ceil(N / 256.0), 1, 1); 


    computeSine<<<dimGrid, dimBlock>>>(d_input, d_output, N);


    cudaDeviceSynchronize();


    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("First 5 sine values:\n");
    for (int i = 0; i < 5; i++) {
        printf("sin(%.2f) = %.6f\n", input[i], output[i]);
    }


    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
