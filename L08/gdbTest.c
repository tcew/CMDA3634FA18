#include <math.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct{

  double x, y, z;

}point_t;

double norm(point_t p){

  double normp = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);

  return normp;
  
}

int main(){

  int N = 100;
  point_t* pt_v;

  // allocate space for N point_t type structs on the "heap"
  pt_v = (point_t*) malloc(N*sizeof(point_t));

  // option 1 for accessing member variables of a point in this array
  pt_v[2].x = 3.2;
  pt_v[2].y = 1.6;
  pt_v[2].z = 0.7;

  // option 2: go straight from pointer to member variable
  (pt_v + 2)->x = 3.2;
  (pt_v + 2)->y = 1.6;
  (pt_v + 2)->z = 0.7;
  
  //  pt_v[100].x = 9.2;

  //  double a = pt_v[200].x;

  pt_v[-100].x = 10.6;
  
  // when we are done with the array we should free it
  free(pt_v);
  pt_v = NULL;

  point_t p;
  p.x = 3;
  p.y = 4;
  p.z = 5;

  double normp = norm(p);

  printf("normp = %lg\n", normp);
  
  return 0;

}
