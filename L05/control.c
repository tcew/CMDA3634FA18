
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){

  float a = 1.;
  
  if(a==1.) {
    printf("a == 1\n");
  }

  float b = 1.f/3.f;
  float c = 2.f/3.f;

  if(b+c==1.f){
    printf("b+c == 1\n");
  }

  int n;
  
  int total = 0;
  // loop 100 times
  for(n=1;n<=100;n=n+1){

    // skip the loop body if n==13
    if(n==13){
      continue;
    }

    total += n;

    // if condition met, stop looping
    if(total > 10000){
      break;
    }
  }
  
  int j = 1;

  // keep iterating
  do{

    j = j*8.5/4.2;

  }while(j<10); // until this fails

  printf("j=%d\n", j);

  return 0;
  
  
}
