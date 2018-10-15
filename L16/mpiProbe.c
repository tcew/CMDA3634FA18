#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/* to compile: 
   
   mpicc -o mpiProbe mpiProbe.c

   to execute with 10 nodes 
   
   mpiexec -n 10 ./mpiProbe

*/


// simple code to explore the use of dynamic message passing
// sequence:
// 1. use MPI_Isend to initiate a message send
// 2. post a blocking MPI_Probe to get metadata from inbound message
// 3. read message status info from MPI_Probe
// 4. create an array to receive inbound message
// 5. post MPI_Recv for inbound message
// 6. use MPI_Wait to block until outgoing message has left

int main(int argc, char **argv){
  
  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Status messageStatus;
  srand48(rank);

  int messageSendCount = 40000 + (drand48()*100); 
  int *messageSendBuffer = (int*) calloc(messageSendCount, sizeof(int));
  int messageSendTag = 999;
  int messageSendDest;

  for(int n=0;n<messageSendCount;++n){
    messageSendBuffer[n] = n;
  }

  messageSendDest = (rank+1)%size;

  // does not return until incoming message copied out of buffer
  MPI_Request messageSendRequest;
  MPI_Isend(messageSendBuffer, messageSendCount, MPI_INT, 
	    messageSendDest, messageSendTag,
	    MPI_COMM_WORLD, &messageSendRequest);

  // now block until any message precursor arrives, with any tag
  MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &messageStatus);

  // read count from messageStatus
  int messageRecvCount;
  MPI_Get_count(&messageStatus, MPI_INT, &messageRecvCount);

  // read tag from messageStatus
  int messageRecvTag = messageStatus.MPI_TAG;

  // read source from messageStatus
  int messageRecvSource = messageStatus.MPI_SOURCE;

  // print out message metadata
  printf("Expecting message with %d ints from rank %d with tag %d\n", 
	 messageRecvCount, messageRecvSource, messageRecvTag);

  // allocate space for inbound message
  int *messageRecvBuffer = (int*) malloc(messageRecvCount*sizeof(int));

  // receive message
  MPI_Recv(messageRecvBuffer, messageRecvCount, MPI_INT, 
	   messageRecvSource, messageRecvTag, MPI_COMM_WORLD, &messageStatus);

  // wait for outgoing message to clear buffer
  MPI_Wait(&messageSendRequest, &messageStatus);
  

  MPI_Finalize();
  return 0;

}
	   
