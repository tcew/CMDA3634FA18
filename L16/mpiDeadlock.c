
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char **argv){
  
  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Status messageStatus;
  int messageCount = atoi(argv[1]);
  int *messageBuffer = (int*) calloc(messageCount, sizeof(int));
  int messageTag = 999;
  int messageDest;
  int messageSource;

  for(int n=0;n<messageCount;++n){
    messageBuffer[n] = n;
  }

  messageDest = (rank+1)%size;
  if(rank>0)
    messageSource = (rank-1);
  else
    messageSource = size-1;

  // does not return until incoming message copied out of buffer
  MPI_Send(messageBuffer, messageCount, MPI_INT, 
	   messageDest, messageTag, MPI_COMM_WORLD);

  // does not return until incoming message copied into buffer
  MPI_Recv(messageBuffer, messageCount, MPI_INT, 
	   messageSource, messageTag, MPI_COMM_WORLD, &messageStatus);
  
  MPI_Finalize();
  return 0;
}
