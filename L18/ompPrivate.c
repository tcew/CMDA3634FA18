
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char **argv){
  int threadCount = 25;

  omp_set_num_threads(threadCount);

  // still serial here
  int i = 6;
  int j = 5;
  
  // fork the program
#pragma omp parallel firstprivate(i) private(j)
  {
    // stuff in this scope gets executed by all OpenMP threads
    int rank = omp_get_thread_num();
    int size = omp_get_num_threads();

    printf("thread %d out of %d threads reports: i=%d, j=%d\n",
	   rank, size, i, j);

  }

  exit(0);
  return 0;

}
