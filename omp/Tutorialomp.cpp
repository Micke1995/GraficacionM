#include <omp.h>
#include <stdio.h>
static long num_steps = 100000; double step;
#define NUM_THREADS 4
int main ()
{ double pi; step = 1.0/(double) num_steps;
 omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
int i, id,nthrds,nthreads; double x, sum;
 id = omp_get_thread_num();
 nthrds = omp_get_num_threads();
 nthreads=nthrds;

 for (i=id, sum=0.0;i< num_steps; i=i+nthreads){
 x = (i+0.5)*step;
 sum += 4.0/(1.0+x*x);
  //printf("%d",nthreads);
 }
 sum = sum*step;

 //
 printf("%d,%d,%d\t",id,nthrds,nthreads);

}
 printf("Este es pi=%f",pi);
}
