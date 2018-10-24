
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char **argv){
  int threadCount = 10;

  omp_set_num_threads(threadCount);

  // still serial here
  int k = 0;
  
  // fork the program
#pragma omp parallel 
  {
    // stuff in this scope gets executed by all OpenMP threads
    int rank = omp_get_thread_num();
    int size = omp_get_num_threads();

    int i = k;
    
    printf("before: thread %d out of %d threads reports: k=%d\n",
	   rank, size, i);

    k = i+1;

    printf("after: thread %d out of %d threads reports: k=%d\n",
	   rank, size, k);
    
  }

  printf("final k = %d\n", k);
  
  exit(0);
  return 0;

}
