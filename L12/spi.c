#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){

  long long int Ninside = 0; // number of random points inside 1/4 circle
  long long int Ntests = 1000000000;
  long long n;
  
  double estpi = 0;

  srand48(12345);
  
  for(n=0;n<Ntests;++n){
    double x = drand48();
    double y = drand48();
    
    if(x*x+y*y<1){
      ++Ninside;
    }
  }

  estpi = 4.*(Ninside/(double)Ntests);

  printf("estPi = %lf\n", estpi);

  return 0;
}
