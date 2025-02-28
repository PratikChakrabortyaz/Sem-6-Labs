#include <mpi.h>
#include <stdio.h>

int factorial(int num)
{
    if (num == 0)
    {
        return 1;
    }
    return num * factorial(num - 1);
}

int main(int argc, char *argv[])
{
    int rank, size, N, A[10], B[10], c, i, sum = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        N = size;
        printf("Enter %d values:\n", N);
        for (i = 0; i < N; i++)
        {
            scanf("%d", &A[i]);
        }
    }

    MPI_Scatter(A, 1, MPI_INT, &c, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("My rank is %d and I have received value %d\n", rank, c);
    c = factorial(c);

    MPI_Gather(&c, 1, MPI_INT, B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("The result gathered in the root:\n");
        for (i = 0; i < N; i++)
        {
            printf("My rank is %d and factorial is %d\n", i, B[i]);
            sum += B[i]; // Sum the factorials
        }
        printf("The sum of all factorials is: %d\n", sum);
    }

    MPI_Finalize();
    return 0;
}
