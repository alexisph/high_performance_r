#include <R.h>
#include <Rinternals.h>
#include <omp.h>
#define CSTACK_DEFNS 7
#include "Rinterface.h"
SEXP dumbCumsum(SEXP a,SEXP reqCores){
  R_CStackLimit=(uintptr_t)-1;
  //Set the number of threads
  PROTECT(reqCores=coerceVector(reqCores,INTSXP));
  int useCores=INTEGER(reqCores)[0];
  int haveCores=omp_get_num_procs();
  if(useCores<=0 || useCores>haveCores) useCores=haveCores;
  omp_set_num_threads(useCores);
  //Do the job
  SEXP ans;
  PROTECT(a=coerceVector(a,REALSXP));
  PROTECT(ans=allocVector(REALSXP,length(a)));
  double* Ans=REAL(ans);
  double* A=REAL(a);
#pragma omp parallel for
  for(int e=0;e<length(a);e++){
    Ans[e]=0.;
    for(int ee=0;ee<e+1;ee++)
      Ans[e]+=A[ee];
  }
  UNPROTECT(3);
  return(ans);
}
