#include<mpi.h>
#include<stdio.h>
#include<ctype.h>
#include<string.h>
#define MAX_M 100

int cnt_non_vowels(char *str,int len)
{
	int cnt=0;
	for(int i=0;i<len;i++)
	{
		char ch=tolower(str[i]);
		if(ch!='a' && ch!='e' && ch!='i' && ch!='o' && ch!='u')
		{
			cnt+=1;
		}
	}
	return cnt;
}

int main(int argc, char *argv[])
{
	int rank,size,N,str_length;
	int loc_cnt; int tot_cnt=0;int chunk_size;
	char str1[MAX_M],loc_str[MAX_M];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	N=size;
	if (rank==0)
	{
		printf("Enter a string:");
		scanf("%s",str1);
		str_length=strlen(str1);
		printf("\n");
		
	}
	MPI_Bcast(&str_length,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(str1,str_length,MPI_CHAR,0,MPI_COMM_WORLD);
	chunk_size=str_length/N; 
	MPI_Scatter(str1,chunk_size,MPI_CHAR,loc_str,chunk_size,MPI_CHAR,0,MPI_COMM_WORLD);
	loc_cnt=cnt_non_vowels(loc_str,chunk_size);

	int cnts[N];
	MPI_Gather(&loc_cnt,1,MPI_INT,cnts,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank==0)
	{
		for(int i=0;i<N;i++)
		{
			tot_cnt+=cnts[i];
			printf("My rank is %d and count is %d\n",i,cnts[i]);
		}
		printf("My rank is %d and total count is %d\n",rank,tot_cnt);

	}
	MPI_Finalize();
	return 0;
	

}