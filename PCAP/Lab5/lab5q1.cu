#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

void printFirst5Results(int *arr, int N) {
    printf("First 5 results:\n");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }
    printf("\n");
}

int main() {
    int N = 1024; 
    int *a, *b;
    int *c, *d, *e;
    int *d_a, *d_b, *d_c, *d_d, *d_e; 

    a = new int[N];
    b = new int[N];
    c = new int[N];
    d = new int[N];
    e = new int[N];

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));
    cudaMalloc((void**)&d_d, N * sizeof(int));
    cudaMalloc((void**)&d_e, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    {
        dim3 dimGrid(N,1,1);  
        dim3 dimBlock(1,1,1);     
    
        vectorAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize(); 
        
        cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Output for subpart a) Block size = N:\n");
        printFirst5Results(c, N);
    }


    {
        dim3 dimGrid(1,1,1);  
        dim3 dimBlock(N,1,1);  
        vectorAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_d, N);
        cudaDeviceSynchronize(); 
        
        cudaMemcpy(d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost); 
        printf("Output for subpart b) N Threads:\n");
        printFirst5Results(d, N);
    }

  
    {
        dim3 dimGrid(ceil(N/256.0),1,1);
        dim3 dimBlock(256,1,1);  

        vectorAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_e, N); 
        cudaDeviceSynchronize(); 
        
        cudaMemcpy(e, d_e, N * sizeof(int), cudaMemcpyDeviceToHost); 
        printf("Output for subpart c) Threads per block = 256:\n");
        printFirst5Results(e, N);
    }


    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);

    return 0;
}

