#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){

  long long int Ninside = 0; // number of random points inside 1/4 circle
  long long int Ntests = 100000000;
  long long int n;
  int rank, size;

  double estpi = 0;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // divide tests among size processes
  //  Ntests = Ntests/size;
  
  // each rank seeds the drand48 random generator differently
  srand48(rank);

  // start timing
  double tic = MPI_Wtime();

  // perform tests on this rank
  for(n=0;n<Ntests;++n){
    double x = drand48();
    double y = drand48();
    
    if(x*x+y*y<1){
      ++Ninside;
    }
  }
  
  // all ranks contribute their value of Ninside to a global sum 
  // with result ending on rank = messageDest = 0
  {
    int messageLength = 1;
    int messageDest = 0;

    long long int *messageOut 
      = (long long int*) calloc(messageLength, sizeof(long long int));

    long long int *messageIn 
      = (long long int*) calloc(messageLength, sizeof(long long int));

    messageOut[0] = Ninside;

    MPI_Reduce(messageOut, messageIn, messageLength, MPI_LONG_LONG_INT, MPI_SUM, messageDest, MPI_COMM_WORLD);

    // only rank == messageDest finalizes the sum
    if(rank==messageDest){
      estpi = 4.*(messageIn[0]/(double)(size*Ntests)); // total number of tests = size * Ntests per rank
    }
  }

  double toc = MPI_Wtime();

  // only rank zero prints out estimate for pi
  if(rank==0){
    //    printf("estPi = %17.15lf\n", estpi);
    printf("%d %lg \n", size, toc-tic);
  }

  MPI_Finalize();

  return 0;
}
