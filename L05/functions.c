#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {

  double x, y;

}vector_t;

vector_t vectorCreate(double a, double b){

  vector_t v;
  v.x = a;
  v.y = b;

  return v;
}

vector_t vectorNormalize(vector_t a){

  double norma = sqrt(a.x*a.x+a.y*a.y);
  vector_t v = vectorCreate(a.x/norma, a.y/norma);

  return v;
}

void vectorPrint(vector_t v){
  printf("vector: %g,%g\n", v.x, v.y);
}

int main(int argc, char **argv){
  
  vector_t n = vectorCreate(0.1, 0.3);
  vector_t a = vectorCreate(0.2, 0.6);

  n = vectorNormalize(n);
  
  vectorPrint(n);

  
  return 0;
}
