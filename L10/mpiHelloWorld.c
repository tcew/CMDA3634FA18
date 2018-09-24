#include <stdio.h>
#include <mpi.h>

/* to compile:

 mpicc -o mpiHelloWorld mpiHelloWorld.c

*/

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  printf("Hello world\n");
  

  MPI_Finalize();
  return 0;
}
