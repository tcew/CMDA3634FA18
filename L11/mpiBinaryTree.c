#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* to compile:

 mpicc -o mpiBinaryTree mpiBinaryTree.c

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
  
  // loop over log2(size) rounds
  int alive = size;

  while(alive>1){

    if(rank<alive && rank>=alive/2){
      int messageDest = rank - (alive/2);
      // send
      MPI_Send(messageOut,
	       messageN,
	       MPI_INT,
	       messageDest,
	       messageTag,
	       MPI_COMM_WORLD);

      printf(" rank %d sends message to rank %d\n",
	     rank, messageDest);
    }

    if(rank<alive/2){
      MPI_Status status;
      int messageSource = rank + (alive/2);
      // send
      MPI_Recv(messageIn,
	       messageN,
	       MPI_INT,
	       messageSource,
	       messageTag,
	       MPI_COMM_WORLD,
	       &status);

      printf(" rank %d recvs message from rank %d\n",
	     rank, messageSource);
    }
    
    alive = alive/2;
    
  }

  if(rank==0) printf("hiya\n");


  MPI_Barrier(MPI_COMM_WORLD);

  
  MPI_Finalize();
  return 0;
}
