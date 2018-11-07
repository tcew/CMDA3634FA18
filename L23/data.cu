
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

__global__ void vectorAddKernel(int N, int *c_a, int *c_b, int *c_c){
  
  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;
  int threadCount = blockDim.x;

  int n = threadIndex + threadCount*blockIndex;
  
  // check if n is in [0,N)
  if(n<N)
    c_c[n] = c_a[n] + c_b[n];


}


int main(int argc, char **argv){

  int N = 4097;

  int threadsPerBlock = 32;
  int blocks = (N+threadsPerBlock-1)/threadsPerBlock;
  
  // ON HOST
  int *h_a = (int*) malloc(N*sizeof(int));
  int *h_b = (int*) malloc(N*sizeof(int));
  int *h_c = (int*) malloc(N*sizeof(int));
  
  int n;
  for(n=0;n<N;++n){
    h_a[n] = 1 + n;
    h_b[n] = 1 - n;
  }

  // ON DEVICE
  int *c_a, *c_b, *c_c;

  cudaMalloc(&c_a, N*sizeof(int));
  cudaMalloc(&c_b, N*sizeof(int));
  cudaMalloc(&c_c, N*sizeof(int));
  
  cudaMemcpy(c_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

  // INITIATE KERNEL ON DEVICE
  vectorAddKernel <<< blocks, threadsPerBlock >>> (N, c_a, c_b, c_c);

  // COPY DATA FROM DEVICE TO HOST
  cudaMemcpy(h_c, c_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  // PRINT ENTRIES
  for(n=0;n<5;++n){
    printf("c[%d] = %d\n", n, h_c[n]);
  }

  cudaDeviceSynchronize();
  cudaFree(c_a);
  cudaFree(c_b);
  cudaFree(c_c);
}
