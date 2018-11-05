
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

__global__ void helloWorld(){
  
  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;

  if(threadIndex%2 == 1)
    printf("Hello World from thread %d of block %d\n",
	   threadIndex, blockIndex);
  else{
    printf("hello from the other threads\n");
    if(threadIndex%3)
      printf("hi from the multiples of 3\n");
  }
}


int main(int argc, char **argv){

  int blocks = 4;
  int threadsPerBlock = 3;

  helloWorld <<< blocks, threadsPerBlock >>> ();

  cudaDeviceSynchronize();
}
