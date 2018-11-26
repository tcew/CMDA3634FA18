/* 

To compile:

   nvcc --arch=sm_60 -O3 -o mandelbrot mandelbrot.c -lm

To create an image with 4096 x 4096 pixels 

    ./mandelbrot 4096 4096 

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI);

// Q2a: add include for CUDA header file here:

#define MXITER 1000

typedef struct {
  
  double r;
  double i;
  
}complex_t;

// return iterations before z leaves mandelbrot set for given c
__device__ int testpoint(complex_t c){

  int iter;
  complex_t z;
  double temp;
  
  z = c;
  
  for(iter=0; iter<MXITER; iter++){  
    temp = (z.r*z.r) - (z.i*z.i) + c.r;
    
    z.i = z.r*z.i*2. + c.i;
    z.r = temp;
    
    if((z.r*z.r+z.i*z.i)>4.0){
      return iter;
    }
  }
  return iter; 
}

// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
__global__ void mandelbrotKernel(int Nre, int Nim, complex_t cmin, complex_t dc, float *count){ 

  // Q2c: replace this loop with a CUDA kernel
  //  for(int n=0;n<Nim;++n){
  //    for(int m=0;m<Nre;++m){

  int n = threadIdx.y + blockIdx.y*blockDim.y;
  int m = threadIdx.x + blockIdx.x*blockDim.x;
  if(n<Nim && m<Nre){
    complex_t c;
    
    c.r = cmin.r + dc.r*m;
    c.i = cmin.i + dc.i*n;
    
    count[m+n*Nre] = (float) testpoint(c);
  }
}


int main(int argc, char **argv){

  // to create a 4096x4096 pixel image 
  // usage: ./mandelbrot 4096 4096 

  int Nre = (argc==3) ? atoi(argv[1]): 4096;
  int Nim = (argc==3) ? atoi(argv[2]): 4096;

  // Q2b: set the number of threads per block and the number of blocks here:
  
  // storage for the iteration counts
  float *count;
  count = (float*) malloc(Nre*Nim*sizeof(float));  

  // Parameters for a bounding box for "c" that generates an interesting image
  const float centRe = -.759856, centIm= .125547;
  const float diam  = 0.151579;

  complex_t cmin; 
  complex_t cmax;
  complex_t dc;

  cmin.r = centRe - 0.5*diam;
  cmax.r = centRe + 0.5*diam;
  cmin.i = centIm - 0.5*diam;
  cmax.i = centIm + 0.5*diam;

  //set step sizes
  dc.r = (cmax.r-cmin.r)/(Nre-1);
  dc.i = (cmax.i-cmin.i)/(Nim-1);

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float *c_count;
  cudaMalloc(&c_count, Nre*Nim*sizeof(float));

  int BX = 16, BY = 16;
  int GX = (Nre+BX-1)/BX, GY = (Nim+BY-1)/BY;

  dim3 B(BX,BY,1);
  dim3 G(GX,GY,1);

  // mark event 
  cudaEventRecord(start);

  // compute mandelbrot set
  mandelbrotKernel <<< G, B >>> (Nre, Nim, cmin, dc, c_count); 
  
  // mark event 
  cudaEventRecord(end);

  // copy from the GPU back to the host here
  cudaMemcpy(count, c_count, Nre*Nim*sizeof(float), cudaMemcpyDeviceToHost);

  // print elapsed time
  float elapsed;
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  elapsed /= 1000.;
  printf("elapsed = %f\n", elapsed);

  // output mandelbrot to ppm format image
  printf("Printing mandelbrot.ppm...");
  writeMandelbrot("mandelbrot.ppm", Nre, Nim, count, 0, 80);
  printf("done.\n");

  free(count);

  exit(0);
  return 0;
}  


/* Output data as PPM file */
void saveppm(const char *filename, unsigned char *img, int width, int height){

  /* FILE pointer */
  FILE *f;
  
  /* Open file for writing */
  f = fopen(filename, "wb");
  
  /* PPM header info, including the size of the image */
  fprintf(f, "P6 %d %d %d\n", width, height, 255);

  /* Write the image data to the file - remember 3 byte per pixel */
  fwrite(img, 3, width*height, f);

  /* Make sure you close the file */
  fclose(f);
}



int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI){

  int n, m;

  unsigned char *rgb   = (unsigned char*) calloc(3*width*height, sizeof(unsigned char));
  
  for(n=0;n<height;++n){
    for(m=0;m<width;++m){
      int id = m+n*width;

      int I = (int) (768*sqrt((double)(img[id]-minI)/(maxI-minI)));
      
      // change this to change palette
      if(I<256)      rgb[3*id+2] = 255-I;
      else if(I<512) rgb[3*id+1] = 511-I;
      else if(I<768) rgb[3*id+0] = 767-I;
      else if(I<1024) rgb[3*id+0] = 1023-I;
      else if(I<1536) rgb[3*id+1] = 1535-I;
      else if(I<2048) rgb[3*id+2] = 2047-I;

    }
  }

  saveppm(fileName, rgb, width, height);

  free(rgb);
  
  return 1;
}


