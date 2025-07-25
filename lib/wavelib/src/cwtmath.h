#ifndef CWTMATH_H_
#define CWTMATH_H_

#include "wtmath.h"
#include "hsfft.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*  lb -lower bound, ub - upper bound, w - time or frequency grid (Size N)   */
void nsfft_exec(fft_object obj, fft_data *inp, fft_data *oup, double lb, double ub, double *w); 

double gamma(double x);

int nint(double N);

#ifdef __cplusplus
}
#endif


#endif /* WAVELIB_H_ */
