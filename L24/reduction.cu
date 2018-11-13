
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

#define BLOCKSIZE 256

__global__ void vectorReductionKernel(int N, int *c_a, int *c_suma){

  volatile __shared__ int s_a[BLOCKSIZE];  // size must be specified at compile time
  
  int threadIndex = threadIdx.x;
  int blockIndex  = blockIdx.x;
  int threadCount = blockDim.x;

  int n = threadIndex + threadCount*blockIndex;
  
  // check if n is in [0,N)
  if(n<N){
    s_a[threadIndex] = c_a[n];
  }else{
    s_a[threadIndex] = 0;
  }

  // barrier for all threads in thread block
  __syncthreads(); // make sure 256 values stored in shared

  if(threadIndex<128)
    s_a[threadIndex] += s_a[threadIndex+128];

  __syncthreads();

  if(threadIndex<64)
    s_a[threadIndex] += s_a[threadIndex+64];

  __syncthreads();

  if(threadIndex<32)
    s_a[threadIndex] += s_a[threadIndex+32];

  //  __syncthreads();

  if(threadIndex<16)
    s_a[threadIndex] += s_a[threadIndex+16];

  //  __syncthreads();

  if(threadIndex<8)
    s_a[threadIndex] += s_a[threadIndex+8]; 

  //  __syncthreads();

  if(threadIndex<4)
    s_a[threadIndex] += s_a[threadIndex+4];

  //  __syncthreads();

  if(threadIndex<2)
    s_a[threadIndex] += s_a[threadIndex+2];

  //  __syncthreads();

  if(threadIndex<1)
    s_a[threadIndex] += s_a[threadIndex+1];
  
  if(threadIndex==0)
    c_suma[blockIndex] = s_a[0];
  
}


int main(int argc, char **argv){

  int N = 100000000;

  int threadsPerBlock = BLOCKSIZE;
  int blocks = ( N+threadsPerBlock-1)/threadsPerBlock;
  
  // ON HOST
  int *h_a = (int*) malloc(N*sizeof(int));
  int *h_b = (int*) malloc(blocks*sizeof(int));
  
  int n;
  for(n=0;n<N;++n){
    h_a[n] = 1;
  }

  // ON DEVICE
  int *c_a, *c_b;

  cudaMalloc(&c_a, N*sizeof(int));
  cudaMalloc(&c_b, blocks*sizeof(int));
  
  cudaMemcpy(c_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // INITIATE KERNEL ON DEVICE
  vectorReductionKernel <<< blocks, threadsPerBlock >>> (N, c_a, c_b);

  cudaEventRecord(end);
  
  cudaEventSynchronize(end);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, end);
  
  elapsed /= 1000.f;

  printf("elapsed time = %g\n", elapsed);

  int NbytesRead = N*sizeof(int);
  int NbytesWritten = blocks*sizeof(int);

  float bandwidth = (NbytesRead + NbytesWritten)/elapsed;

  printf("bandwidth %g GB/s\n", bandwidth/1.e9);


  // COPY DATA FROM DEVICE TO HOST
  cudaMemcpy(h_b, c_b, blocks*sizeof(int), cudaMemcpyDeviceToHost);

  int reda = 0;
  for(n=0;n<blocks;++n)
    reda += h_b[n];

  printf("sum entries a is %d\n", reda);

  // PRINT ENTRIES
  for(n=0;n<5;++n){
    printf("suma[%d] = %d\n", n, h_b[n]);
  }

  cudaDeviceSynchronize();
  cudaFree(c_a);
  cudaFree(c_b);
}
