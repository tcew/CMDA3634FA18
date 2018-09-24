#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* to compile:

 mpicc -o mpiHelloWorld mpiHelloWorld.c

*/

int main(int argc, char **argv){

  int rank, size;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  printf("Hello world from %d of %d ranks\n", rank, size);

  int messageN = 10;
  int messageTag = 999;
    
  // rank 0 is going to make a message and send it to rank 1
  if(rank==0) {

    int *messageOut = (int*) malloc(messageN*sizeof(int));
    int messageDest = 1;
    for(int n=0;n<messageN;++n){
      messageOut[n] = n;
    }

    MPI_Send(messageOut,
	     messageN,
	     MPI_INT, // data type from MPI
	     messageDest,
	     messageTag,
	     MPI_COMM_WORLD);
  }

  // rank 1 receives message from rank 0
  if(rank==1) {
    MPI_Status status;
    
    int *messageIn = (int*) malloc(messageN*sizeof(int));
    int messageSource = 0;

    MPI_Recv(messageIn,
	     messageN,
	     MPI_INT, // data type from MPI
	     messageSource,
	     messageTag,
	     MPI_COMM_WORLD,
	     &status);

    for(int n=0;n<messageN;++n){
      printf("rank %d got messageIn[%d] = %d\n",
	     rank, n, messageIn[n]);
    }
    
  }
  
  MPI_Finalize();
  return 0;
}
