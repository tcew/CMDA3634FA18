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

  Ntests = Ntests/size;
  
  srand48(rank);
  

  double tic = MPI_Wtime();

  for(n=0;n<Ntests;++n){
    double x = drand48();
    double y = drand48();
    
    if(x*x+y*y<1){
      ++Ninside;
    }
  }

  double toc = MPI_Wtime();


  {
    int messageLength = 1;
    int messageDest = 0;

    long long int *messageOut 
      = (long long int*) calloc(messageLength, sizeof(long long int));

    long long int *messageIn 
      = (long long int*) calloc(messageLength, sizeof(long long int));

    messageOut[0] = Ninside;

    MPI_Reduce(messageOut, messageIn, messageLength, MPI_LONG_LONG_INT, MPI_SUM, messageDest, MPI_COMM_WORLD);
    
    if(rank==messageDest){
      
      estpi = 4.*(messageIn[0]/(double)(size*Ntests)); // total number of tests = size * Ntests per rank
    }
  }

  
  if(rank==0){
    //    printf("estPi = %17.15lf\n", estpi);
    printf("%d %lg \n", size, toc-tic);
  }

  MPI_Finalize();

  return 0;
}
