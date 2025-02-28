#include<mpi.h>
#include<stdio.h>
int main(int argc,char *argv[])
{
int rank,size,x;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Status status;
if (rank==0)
{
printf("Enter a value in master process:");
scanf("%d",&x);
for(int i=1;i<size;i++)
{
MPI_Send(&x,1,MPI_INT,i,0,MPI_COMM_WORLD);
fprintf(stdout,"My rank is %d and I sent %d to process %d\n",rank,x,i);
fflush(stdout);
}
}
else
{
MPI_Recv(&x,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
fprintf(stdout,"My rank is %d and I received %d\n",rank,x);
fflush(stdout);
}
MPI_Finalize();
return 0;
} 

