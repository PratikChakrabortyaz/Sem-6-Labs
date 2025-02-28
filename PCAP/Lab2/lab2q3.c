#include<mpi.h>
#include<stdio.h>
int main(int argc,char *argv[])
{
int rank,size;
int arr[100];
int recv_value,result;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Status status;
if (rank==0)
{
fprintf(stdout,"Enter %d elements:\n",size);
for(int i=0;i<size;i++)
{
fprintf(stdout,"Element %d:",i+1);
fflush(stdout);
scanf("%d",&arr[i]);
}
for(int i=1;i<size;i++)
{
MPI_Send(&arr[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
fprintf(stdout,"My rank is %d and I sent %d to process %d\n",rank,arr[i],i);
fflush(stdout);
}
}
else
{
MPI_Recv(&recv_value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
fprintf(stdout,"My rank is %d and I received %d\n",rank,recv_value);
fflush(stdout);
if(rank%2==0)
{
result=recv_value*recv_value;
fprintf(stdout,"My rank is %d and new value is %d\n",rank,result);
fflush(stdout);
}
else
{
result=recv_value*recv_value*recv_value;
fprintf(stdout,"My rank is %d and new value is %d\n",rank,result);
fflush(stdout);
}
}
MPI_Finalize();
return 0;
}

