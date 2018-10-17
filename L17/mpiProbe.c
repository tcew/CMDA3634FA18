

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){

  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  srand48(rank);

  int messageCount = 100 + 200*(drand48());

  int *messageSendBuffer = (int*) 
    malloc(messageCount*sizeof(int));

  for(int n=0;n<messageCount;++n){
    messageSendBuffer[n] = n;
  }

  int messageDest = (rank+1)%size;
  int messageTag = 999;
  
  MPI_Request messageSendRequest;
  MPI_Request messageRecvRequest;

  MPI_Isend(messageSendBuffer,
	    messageCount,
	    MPI_INT,
	    messageDest, 
	    messageTag, 
	    MPI_COMM_WORLD,
	    &messageSendRequest);


  int messageRecvCount;

  MPI_Status status;

  // block until next message is in bound
  MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

  // find out where the message came from
  int messageRecvSource = status.MPI_SOURCE;

  // find out what the message tag is
  int messageRecvTag = status.MPI_TAG;

  // how many entries in message
  MPI_Get_count(&status, MPI_INT, &messageRecvCount);

  int *messageRecvBuffer = (int*) 
    malloc(messageRecvCount*sizeof(int));
  
  MPI_Irecv(messageRecvBuffer,
	    messageRecvCount,
	    MPI_INT,
	    messageRecvSource, 
	    messageRecvTag, 
	    MPI_COMM_WORLD, 
	    &messageRecvRequest);


  MPI_Wait(&messageSendRequest, &status);
  MPI_Wait(&messageRecvRequest, &status);

  printf("after wait: messageRecvBuffer[%d] = %d\n", 
	 3, messageRecvBuffer[3]);


  MPI_Finalize();
  return 0;

}
