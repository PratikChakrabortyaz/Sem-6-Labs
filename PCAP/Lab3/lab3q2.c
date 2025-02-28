#include<mpi.h>
#include<stdio.h>
#define MAX_M 100

int main(int argc,char *argv[])
{
	int rank,size,N,M;
	int data[MAX_M*MAX_M];
	int ind_data[MAX_M];
	int sub_sum=0;int tot_sum=0;
	float sub_avg=0; float tot_avg;float all_avg[MAX_M];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	N=size;
	if (rank==0)
	{
		printf("Enter value of M: ");
		scanf("%d",&M);
	}
	MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank==0)
	{
		printf("Enter %d integers \n",N*M);
		for(int i=0;i<N*M;i++)
		{
			scanf("%d",&data[i]);
		}
	}
	MPI_Scatter(data,M,MPI_INT,ind_data,M,MPI_INT,0,MPI_COMM_WORLD);
	for(int i=0;i<M;i++)
	{
		sub_sum+=ind_data[i];
	}
	sub_avg=(float)sub_sum/M;
	printf("My rank is %d and I have avg value of %f \n",rank,sub_avg);

	MPI_Gather(&sub_avg,1,MPI_FLOAT,all_avg,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	if(rank==0)
	{
		for(int i=0;i<N;i++)
		{
			tot_sum+=all_avg[i];
		}
		tot_avg=(float)tot_sum/N;
		printf("My rank is %d and total avg is %f \n",rank,tot_avg);
	}
	MPI_Finalize();
	return 0;

}