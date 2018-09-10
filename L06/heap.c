#include <stdio.h>
#include <stdlib.h>

int main(){

  int N = 10;

  // get pointer to N integers located in the dynamically allocated heap
  int *pt_v = (int*) malloc(N*sizeof(int));

  // set first entry of array on heap to 6
  pt_v[0] = 6;

  // set second entry of array on heap to 3
  pt_v[1] = 3;
  pt_v[2] = 2;

  // set 11th entry to 7 - but we only requested 10 integers
  //  pt_v[10] = 7;

  int *pt_a = (int*) malloc(N*sizeof(int));
  int *pt_b = (int*) malloc(N*sizeof(int));
  int *pt_c = (int*) malloc(N*sizeof(int));

  for(int n=0;n<N;++n){
    pt_a[n] = 1+n;
    pt_b[n] = 1-n;
  }

  for(int n=0;n<N;++n){
    pt_c[n] = pt_a[n] + pt_b[n];
  }

  for(int n=0;n<N;++n){
    printf("c[%d] = %d\n", n, pt_c[n]);
  }
    
  return 0;
}
  
