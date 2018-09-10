#include <stdio.h>
#include <stdlib.h>

void function(int* pt_i, int* pt_j, int* pt_k){

  *pt_j = 2*(*pt_i);
  *pt_k = 3*(*pt_i);
  
}

int main(){

  int i = 6;
  int j = 4;

  // get pointer to i (address of i in memory)
  int* pt_i = &i;
  int* pt_j = &j;

  // print pointer in hexadecimal
  printf("pt_i = %p\n", pt_i);
  printf("pt_j = %p\n", pt_j);

  // print pointer as a 64 bit unsigned integer
  printf("pt_i = %llu\n", (long long unsigned int) pt_i);
  printf("pt_j = %llu\n", (long long unsigned int) pt_j);
  
  // pointer arithmetic
  printf("pt_j - pt_i = %llu\n",
	 (long long unsigned) pt_j
	 -(long long unsigned) pt_i);  
  // set value of i to 7 by using the address of i (the pointer to i)
  *pt_i = 7;

  // a buffer overflow error
  *(pt_j-1) = 2; 
  
  printf("i=%d\n", i);
  printf("j=%d\n", j);

  int k;
  i = 1;
  j = 0;
  k = 0;

  function(&i,&j,&k);


  
  printf("i=%d, j=%d, k=%d\n", i, j, k);	
  return 0;
}
  
