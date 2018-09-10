
#include <stdio.h>
#include <stdlib.h>

int main(){

  int i = 6;
  int j = 4;
  
  int* pt_i = &i;

  printf("pt_i = %p\n", pt_i);

  printf("pt_i = %llu\n", (long long unsigned int) pt_i);

  *pt_i = 7;

  printf("i=%d\n", i);
  
  return 0;
}
  
