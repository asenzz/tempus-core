//
// Created by ban on 5/31/18.
//

#ifndef SVR_KERNEL_GLOBAL_ALIGNMENT_H
#define SVR_KERNEL_GLOBAL_ALIGNMENT_H

//#include <stdlib.h>
#include <cmath>
/* Useful constants */


#define LOG0 -10000          /* log(0) */
#define LOGP(x, y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))


#include "svm.h"
#include <vector>

void svm_node_to_vector(const svm_node *_x, std::vector<double> &x)
{
    int i = 0;
    while (_x->index != -1) {
        x.push_back(_x->value);
        _x++;
        ++i;
    }
}

double logGAK(const svm_node *x, const svm_node *y, int nX, int nY, int dimvect, double sigma, double triangular)
{
    std::vector<double> seq1;
    svm_node_to_vector(x, seq1);
    std::vector<double> seq2;
    svm_node_to_vector(y, seq2);
    int i, j, ii, cur, old, curpos, frompos1, frompos2, frompos3;
    double aux;
    int cl = nY + 1;                /* length of a column for the dynamic programming */


    double sum = 0;
    double gram, sig;
    /* log_m is the array that will stores two successive columns of the (nX+1) x (nY+1) table used to compute the final kernel value*/
    std::vector<double> log_m(2 * cl);
    //double * log_m = malloc(2*cl * sizeof(double));

    int trimax = (nX > nY) ? nX - 1 : nY - 1; /* Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY */

    //double *log_triangular_coefs = malloc((trimax+1) * sizeof(double));
    std::vector<double> log_triangular_coefs(trimax + 1);
    if (triangular > 0) {
        /* initialize */
        for (i = 0; i <= trimax; i++) {
            log_triangular_coefs[i] = LOG0; /* Set all to zero */
        }

        for (i = 0; i < ((trimax < triangular) ? trimax + 1 : triangular); i++) {
            log_triangular_coefs[i] = log(1 - i / triangular);
        }
    } else
        for (i = 0; i <= trimax; i++) {
            log_triangular_coefs[i] = 0; /* 1 for all if triangular==0, that is a log value of 0 */
        }
    sig = -1 / (2 * sigma * sigma);



    /****************************************************/
    /* First iteration : initialization of columns to 0 */
    /****************************************************/
    /* The left most column is all zeros... */
    for (j = 1; j < cl; j++) {
        log_m[j] = LOG0;
    }
    /* ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0. */
    log_m[0] = 0;

    /* Cur and Old keep track of which column is the current one and which one is the already computed one.*/
    cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
    old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */

    /************************************************/
    /* Next iterations : processing columns 1 .. nX */
    /************************************************/

    /* Main loop to vary the position for i=1..nX */
    curpos = 0;
    for (i = 1; i <= nX; i++) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur * cl;                  /* index of the state (i,0) */
        log_m[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (j = 1; j <= nY; j++) {
            curpos = cur * cl + j;            /* index of the state (i,j) */
            if (log_triangular_coefs[abs(i - j)] > LOG0) {
                frompos1 = old * cl + j;            /* index of the state (i-1,j) */
                frompos2 = cur * cl + j - 1;          /* index of the state (i,j-1) */
                frompos3 = old * cl + j - 1;          /* index of the state (i-1,j-1) */

                /* We first compute the kernel value */
                sum = 0;
                for (ii = 0; ii < dimvect; ii++) {
                    sum += (seq1[i - 1 + ii * nX] - seq2[j - 1 + ii * nY]) *
                           (seq1[i - 1 + ii * nX] - seq2[j - 1 + ii * nY]);
                }
                gram = log_triangular_coefs[abs(i - j)] + sum * sig;
                gram -= log(2 - exp(gram));

                /* Doing the updates now, in two steps. */
                aux = LOGP(log_m[frompos1], log_m[frompos2]);
                log_m[curpos] = LOGP(aux, log_m[frompos3]) + gram;
            } else {
                log_m[curpos] = LOG0;
            }
        }
        /* Update the culumn order */
        cur = 1 - cur;
        old = 1 - old;
    }
    aux = log_m[curpos];
    //free(log_m);
    //free(log_triangular_coefs);
    /* Return the logarithm of the Global Alignment Kernel */
    return aux;

}


#endif //SVR_KERNEL_GLOBAL_ALIGNMENT_H
