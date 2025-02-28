#include<mpi.h>
#include<stdio.h>
int main(int argc, char *argv[])
{
int rank,size;
char s[]="HELLO";
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
if (rank < sizeof(s) - 1) {  
        if (s[rank] >= 'A' && s[rank] <= 'Z') {
            s[rank] = s[rank] + 32;  
        } else if (s[rank] >= 'a' && s[rank] <= 'z') {
            s[rank] = s[rank] - 32;  
        }
    }

    printf("Rank %d: Toggled character '%c' at position %d\n", rank, s[rank], rank);

    MPI_Finalize();  
    return 0;
}
