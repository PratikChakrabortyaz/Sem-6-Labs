#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    long long local_fact = 1, global_sum = 0, cumulative_fact = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N = size;  

    local_fact = rank + 1; 

    MPI_Scan(&local_fact, &cumulative_fact, 1, MPI_LONG_LONG, MPI_PROD, MPI_COMM_WORLD);

    printf("My rank is %d and factorial is %d\n", rank,cumulative_fact);

    MPI_Reduce(&cumulative_fact, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final sum of factorials from 1! to %d! is: %lld\n", N, global_sum);
    }

    MPI_Finalize();
    return 0;
}

