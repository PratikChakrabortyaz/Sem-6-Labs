#include <stdio.h>
#include <cuda.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

__global__ void transformString(char *S, char *RS, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        int start_idx = (len * idx) - (idx * (idx - 1)) / 2;  
        for (int j = 0; j < len - idx; j++) {
            RS[start_idx + j] = S[j];
        }
    }
}

int main() {
    char h_S[100], *d_S, *d_RS, h_RS[300];
    
    printf("Enter the input string S: ");
    scanf("%s", h_S);
    
    int len = strlen(h_S);
    int rs_len = (len * (len + 1)) / 2; 
    
    cudaMalloc((void**)&d_S, len * sizeof(char));
    cudaMalloc((void**)&d_RS, rs_len * sizeof(char));
    
    cudaMemcpy(d_S, h_S, len * sizeof(char), cudaMemcpyHostToDevice);
    
    dim3 dimGrid((int)ceil((float)len / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);
    
    transformString<<<dimGrid, dimBlock>>>(d_S, d_RS, len);
    
    cudaMemcpy(h_RS, d_RS, rs_len * sizeof(char), cudaMemcpyDeviceToHost);
    h_RS[rs_len] = '\0'; 
    
    printf("Output String RS: %s\n", h_RS);
    
    cudaFree(d_S);
    cudaFree(d_RS);
    
    return 0;
}
