

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){

  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int messageCount = atoi(argv[1]);

  int *messageSendBuffer = (int*) 
    malloc(messageCount*sizeof(int));

  int *messageRecvBuffer = (int*) 
    malloc(messageCount*sizeof(int));
  
  for(int n=0;n<messageCount;++n){
    messageSendBuffer[n] = n;
  }

  int messageDest = (rank+1)%size;
  int messageSource;
  if(rank>0) messageSource = rank-1;
  else messageSource = size-1;
  
  int messageTag = 999;
  
  MPI_Request messageSendRequest;
  MPI_Request messageRecvRequest;

  MPI_Irecv(messageRecvBuffer,
	    messageCount,
	    MPI_INT,
	    messageSource, 
	    messageTag, 
	    MPI_COMM_WORLD, 
	    &messageRecvRequest);



  MPI_Isend(messageSendBuffer,
	    messageCount,
	    MPI_INT,
	    messageDest, 
	    messageTag, 
	    MPI_COMM_WORLD,
	    &messageSendRequest);

  printf("before wait: messageRecvBuffer[%d] = %d\n", 
	 3, messageRecvBuffer[3]);
  
  int bah = rank;
  for(int foo=0;foo<10000;++foo){
    bah *= foo*bah - 6;
  }
  printf("bah = %d\n", bah);
  MPI_Status status;


  MPI_Wait(&messageSendRequest, &status);
  MPI_Wait(&messageRecvRequest, &status);



  printf("before wait: messageRecvBuffer[%d] = %d\n", 
	 3, messageRecvBuffer[3]);


  MPI_Finalize();
  return 0;

}
