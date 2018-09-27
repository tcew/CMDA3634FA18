#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* to compile:

 mpicc -o mpiBinaryTreeNonPowerOfTwo mpiBinaryTreeNonPowerOfTwo.c

*/

int main(int argc, char **argv){

  int rank, size;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int messageN = 1;
  int messageTag = 999;

  int *messageOut = (int*) malloc(messageN*sizeof(int));
  int *messageIn  = (int*) malloc(messageN*sizeof(int));

  messageOut[0] = rank;
  
  // loop over log2(size) rounds
  int alive = size;

  while(alive>1){
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank<alive && rank>=(alive+1)/2){
      int messageDest = rank - ((alive+1)/2);

      printf(" rank %d sends message to rank %d\n",
	     rank, messageDest);

      
      // send
      MPI_Send(messageOut,
	       messageN,
	       MPI_INT,
	       messageDest,
	       messageTag,
	       MPI_COMM_WORLD);

    }

    if(rank<(alive+1)/2){
      MPI_Status status;
      int messageSource = rank + ((alive+1)/2);
      if(messageSource<size && messageSource<alive){
	// send
	MPI_Recv(messageIn,
		 messageN,
		 MPI_INT,
		 messageSource,
		 messageTag,
		 MPI_COMM_WORLD,
		 &status);
      printf(" rank %d recvs message %d from rank %d\n",
	     rank, messageIn[0], messageSource);
      }
    }
    
    alive = (alive+1)/2;
    
  }

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(rank==0) printf("hiya\n");

  MPI_Finalize();
  return 0;
}
