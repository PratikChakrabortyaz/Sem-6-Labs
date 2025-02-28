#include<mpi.h>
#include<stdio.h>
#include<math.h>
int main(int argc, char *argv[])
{
int rank,size;
int x=2;int y=5;
int result;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
switch(rank)
{
case 0:
result=x+y;
printf("My rank is %d and result is %d \n",rank,result);
break;
case 1:
result=x-y;
printf("My rank is %d and result is %d \n",rank,result);
break; 
case 2:
result=x*y;
printf("My rank is %d and result is %d \n",rank,result);
break;
case 3:
result=x/y;
printf("My rank is %d and result is %d \n",rank,result);
break;
}
MPI_Finalize();
return 0;
}
