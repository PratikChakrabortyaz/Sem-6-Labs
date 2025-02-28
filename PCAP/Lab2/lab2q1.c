#include<mpi.h>
#include<stdio.h>
#include<ctype.h>
#include<string.h>
void toggle_case(char* str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (isupper(str[i])) {
            str[i] = tolower(str[i]);
        } else if (islower(str[i])) {
            str[i] = toupper(str[i]);
        }
    }
}
int main(int argc,char *argv[])
{
int rank,size,x;char s[100];int len;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Status status;
if (rank==0)
{
printf("Enter a word to send:\n");
scanf("%s",s);
len=strlen(s);
MPI_Ssend(&len,1,MPI_INT,1,0,MPI_COMM_WORLD);
MPI_Ssend(s,len,MPI_CHAR,1,1,MPI_COMM_WORLD);
fprintf(stdout,"My rank is %d and I send %s\n",rank,s);
fflush(stdout);
MPI_Recv(s,len,MPI_CHAR,1,2,MPI_COMM_WORLD,&status);
fprintf(stdout,"My rank is %d and I recived toggled word is %s\n",rank,s);
fflush(stdout);
}
else
{
MPI_Recv(&len,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
printf("Length is %d\n",len);
MPI_Recv(s,len,MPI_CHAR,0,1,MPI_COMM_WORLD,&status);
fprintf(stdout,"My rank is %d and I received %s\n",rank,s);
toggle_case(s);
fprintf(stdout,"My rank is %d and toggled word is %s\n",rank,s);
fflush(stdout);
MPI_Ssend(s,len,MPI_CHAR,0,2,MPI_COMM_WORLD);
}
MPI_Finalize();
return 0;
}


