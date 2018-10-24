


#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv){

  int n, i, j, k;
  int N = 100;

  int *v = (int*) malloc(N*sizeof(int));
  int *w = (int*) malloc(N*sizeof(int));

  // globally set default number of threads
  omp_set_num_threads(20);
  
  // Q1: can I use an opemp directive to make this parallel without changing the outcome
  // [ zero entries in array with zero ]
#pragma omp parallel for
  for(n=0;n<N;++n){
    v[n] = 0;
  }

  // Q2: can I use an opemp directive to make this parallel without changing the outcome
  // [ fill array with values 0:N-1 ]
#pragma omp parallel for
  for(n=0;n<N;++n){
    v[n] = n;
  }

  // Q3: can I use an opemp directive to make this parallel without changing the outcome
  // [ fill arrays with values 0:N-1 in reverse order ]
#pragma omp parallel for
  for(n=N-1;n>=0;--n){
    v[n] = n;
  }


  // Q4: how can I use an opemp directive to make this parallel without changing the outcome
  // [ propagate value from v[0] to all entries of v ]
#pragma omp parallel for
  for(n=1;n<N;++n){
    v[n] = v[0];
  }


  v[0] = 1;

  // Q5: can I use an opemp directive to make this parallel without changing the outcome
  // [ store numbers 1 to N in array ]
  // NOT  A GOOD CANDIDATE BECAUSE OF ORDER DEPENDENCE OF
  // ITERATIONS
  for(n=1;n<N;++n){
    v[n] = v[n-1] + 1;
  }

  v[0] = 0;

  // Q6: how can I use an opemp directive to make this parallel without changing the outcome
  // [ store v in reverse order in w ]
  // We have to give each thread a private variable i
#pragma omp parallel for private(i)
  for(n=1;n<N;++n){
    i = N-1-n;
    w[n] = v[i];
  }

  // Q7: how can I use an opemp directive to make this parallel without changing the outcome
  // [ reverse entries in an array ]
#pragma omp parallel for private(i,j)
  for(n=0;n<N/2;++n){ 
    i = N-1-n;
    j = v[n];
    v[n] = v[i];
    v[i] = j;
  }


  // Q8: can I use an opemp directive to make this parallel without changing the outcome
  // [ reverse entries in an array ]
#pragma omp parallel for
  for(n=0;n<N/2;++n){ 
    int m = N-1-n;
    int tmp = v[n];
    v[n] = v[m];
    v[m] = tmp;
  }


  i = 0;
  // SOME LOOP CARRIED DEPENDENCE IS OK
  // IN PARTICULAR IF WE CAN TURN THE ACCUMULATOR
  // INTO A REDUCTION VARIABLE
#pragma omp parallel for reduction(+:i)
  for(n=0;n<N;++n){
    i = i+n;
  }
    
}

  
