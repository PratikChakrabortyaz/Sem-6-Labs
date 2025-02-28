#include<mpi.h>
#include<stdio.h>
int factorial(int num)
{
if (num==0)
{
return 1;
}
return num*factorial(num-1);
}
int fibonacci(int num) 
{
if (num==0)
{
return 0;
}
if (num==1)
{
return 1;
}
return fibonacci(num-1)+fibonacci(num-2);
}
int main(int argc, char *argv[])
{
int rank,size,result;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
if (rank%2==0)
{
result=factorial(rank);
printf("My rank is %d and factorial is %d\n",rank,result);
}
else
{
result=fibonacci(rank);
printf("My rank is %d and fibonacci number is %d\n",rank,result);
}

MPI_Finalize();  
return 0;
}
