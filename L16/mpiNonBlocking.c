
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/* to compile: 
   
   mpicc -o mpiNonBlocking mpiNonBlocking.c

   to execute with 10 nodes and messages of length 20: 
   
   mpiexec -n 10 ./mpiNonBlocking 20

*/

int main(int argc, char **argv){
  
  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Status messageStatus;
  int messageCount = atoi(argv[1]);
  int *messageSendBuffer = (int*) calloc(messageCount, sizeof(int));
  int *messageRecvBuffer = (int*) calloc(messageCount, sizeof(int));
  int messageTag = 999;
  int messageDest;
  int messageSource;

  for(int n=0;n<messageCount;++n){
    messageSendBuffer[n] = n;
  }

  messageDest = (rank+1)%size;
  if(rank>0)
    messageSource = (rank-1);
  else
    messageSource = size-1;

  // returns immediately, potentially before message is copied out of buffer
  MPI_Request messageSendRequest;
  MPI_Isend(messageSendBuffer, messageCount, MPI_INT, 
	    messageDest, messageTag, MPI_COMM_WORLD, &messageSendRequest);

  // returns immediately, potentially before message is copied out of buffer
  MPI_Request messageRecvRequest;
  MPI_Irecv(messageRecvBuffer, messageCount, MPI_INT, 
	    messageSource, messageTag, MPI_COMM_WORLD, &messageRecvRequest);

  int testId = 3;
  printf("before wait: message[%d] = %d\n", testId, messageRecvBuffer[testId]);
  
  // now wait for incoming message to have been copied into buffer
  MPI_Wait(&messageRecvRequest, &messageStatus);

  // now wait for outgoing message to have been copied out of buffer
  MPI_Wait(&messageSendRequest, &messageStatus);
  
  printf("after wait: message[%d] = %d\n", testId, messageRecvBuffer[testId]);

  MPI_Finalize();
  return 0;
}
