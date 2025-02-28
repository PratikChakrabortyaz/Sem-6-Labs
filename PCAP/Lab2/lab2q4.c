#include<mpi.h>
#include<stdio.h>
int main(int argc,char *argv[])
{
int rank,size,x;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Status status;
if(rank==0)
{
fprintf(stdout,"Enter an integer value:");
fflush(stdout);
scanf("%d",&x);
fprintf(stdout,"My rank is %d and I have value %d\n",rank,x);
fflush(stdout);
MPI_Send(&x,1,MPI_INT,1,0,MPI_COMM_WORLD);
MPI_Recv(&x, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD,&status);
fprintf(stdout,"My rank is %d and I have received value %d\n",rank,x);
}
else
{
MPI_Recv(&x,1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&status);
x++;
fprintf(stdout,"My rank is %d and received value %d incremented to %d\n",rank,x-1,x);
fflush(stdout);
if (rank<size-1)
{
MPI_Send(&x,1,MPI_INT,rank+1,0,MPI_COMM_WORLD);
}
else
{
MPI_Send(&x,1,MPI_INT,0,0,MPI_COMM_WORLD);
}
}
MPI_Finalize();
return 0;
}
