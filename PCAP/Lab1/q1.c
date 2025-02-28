#include<mpi.h>
#include<stdio.h>
#include<math.h>
int main(int argc, char *argv[])
{
int rank,size;
int x=3;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
int result=pow(3,rank);
printf("My rank is %d and result is %d \n",rank,result);
MPI_Finalize();
return 0;
}
