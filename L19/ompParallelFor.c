
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv){

  int n;
  int N = 100;
  int *v = (int*) malloc(N*sizeof(int));

  // globally set default number of threads
  omp_set_num_threads(20);

  // print 4 hello worlds
#pragma omp parallel
  {
    printf("Hello World!\n");
  }

  // ok
#pragma omp parallel for  
  for(n=0;n<N;++n){
    v[n] = 0;
  }

  v[0] = 1;
#pragma omp parallel for  
  for(n=1;n<N;++n){

    int oldvn = v[n-1];

    int rank = omp_get_thread_num();
    printf("hello from rank %d, writing to v[%d]\n", rank, n);

    v[n] = oldvn+1;
  }

  for(n=0;n<N;++n){
    printf("v[%d] = %d\n", n, v[n]);
  }
  
#pragma omp parallel
  {
    printf(" hi \n");

#pragma omp for nowait
    for(n=0;n<N;++n){
      v[n] = 0;
    }

    printf(" ya \n");  

  }
  
}
