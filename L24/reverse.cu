
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

__global__ void vectorReverseKernel(int N, int *c_a){
  
  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;
  int threadCount = blockDim.x;

  int n = threadIndex + threadCount*blockIndex;
  
  // check if n is in [0,N)
  if(n<N){
    int tmp = c_a[n];
    c_a[n] = c_a[N-1-n];
    c_a[N-1-n] = tmp;
  }

}


int main(int argc, char **argv){

  int N = 4097;

  int threadsPerBlock = 32;
  int blocks = ( (N/2)+threadsPerBlock-1)/threadsPerBlock;
  
  // ON HOST
  int *h_a = (int*) malloc(N*sizeof(int));
  
  int n;
  for(n=0;n<N;++n){
    h_a[n] = 1 + n;
  }

  // ON DEVICE
  int *c_a;

  cudaMalloc(&c_a, N*sizeof(int));
  
  cudaMemcpy(c_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

  // INITIATE KERNEL ON DEVICE
  vectorReverseKernel <<< blocks, threadsPerBlock >>> (N, c_a);

  // COPY DATA FROM DEVICE TO HOST
  cudaMemcpy(h_a, c_a, N*sizeof(int), cudaMemcpyDeviceToHost);

  // PRINT ENTRIES
  for(n=0;n<5;++n){
    printf("a[%d] = %d\n", n, h_a[n]);
  }

  cudaDeviceSynchronize();
  cudaFree(c_a);
}
