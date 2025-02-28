#include <stdio.h>
#include <cuda.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

__global__ void countWordOccurrences(char *sentence, int sentence_len, char *word, int word_len, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx + word_len <= sentence_len) { 
        int match = 1;
        for (int i = 0; i < word_len; i++) {
            if (sentence[idx + i] != word[i]) {
                match = 0;
                break;
            }
        }
        
        if (match && (idx + word_len == sentence_len || sentence[idx + word_len] == ' ')) {
            atomicAdd(count, 1); 
        }
    }
}

int main() {
    char h_sentence[256], h_word[100];
    int *d_count, h_count = 0;
    char *d_sentence, *d_word;
    
    printf("Enter a sentence: ");
    fgets(h_sentence, 256, stdin);
    
    printf("Enter the word to count: ");
    scanf("%s", h_word);
    
    int sentence_len = strlen(h_sentence);
    int word_length = strlen(h_word);
    
    cudaMalloc((void**)&d_sentence, sentence_len * sizeof(char));
    cudaMalloc((void**)&d_word, word_length * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));
    
    cudaMemcpy(d_sentence, h_sentence, sentence_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, word_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid((int)ceil((float)sentence_len / THREADS_PER_BLOCK));
    dim3 dimBlock(THREADS_PER_BLOCK);
    
    countWordOccurrences<<<dimGrid, dimBlock>>>(d_sentence, sentence_len, d_word, word_length, d_count);
    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("The word '%s' appears %d times in the sentence.\n", h_word, h_count);
    
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    
    return 0;
}
