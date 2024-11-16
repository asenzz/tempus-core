//
// Created by zarko on 3/30/21.
//


#include "nm.hpp"
#include "../../SVRCommon/include/common.hpp"
#include "common/spin_lock.hpp"


namespace svr {
namespace optimizer {

std::pair<double, std::vector<double>>
nm(const svr::optimizer::loss_callback_t &loss_fun, const std::vector<double> &initial_values, const NM_parameters &nm_parameters)
{
    const auto n = initial_values.size(); // number of parameters, i.e. vector length
    std::vector<double> xmin(n);
    double ynewlo;
    size_t ifault{0};
    const std::vector<double> step(n, 1.0);
    const size_t check_for_convergence{4};
    double ccoeff = 0.5;
    double del;
    double dn;
    double dnn;
    double ecoeff{2.0};
    double eps{0.001};
    size_t i;
    int ihi;

    size_t j;
    int jcount;
    int ell;
    size_t nn;
    std::vector<double> p(n * (n + 1));
    std::vector<double> p2star(n);
    std::vector<double> pbar(n);
    std::vector<double> pstar(n);
    double rcoeff{1.0};
    double rq;
    double x;
    std::vector<double> y(n + 1);
    double y2star;

    double ystar;
    double z;

    size_t icount = 0;
    std::vector<double> start = initial_values;
    jcount = check_for_convergence;
    dn = (double) (n);
    nn = n + 1;
    dnn = (double) (nn);
    del = 1.0;
    rq = nm_parameters.tolerance_ * dn;
    //
    //  Initial or restarted loop.
    //
    LOG4_INFO("NM: initialization");

    while (true) {
        for (size_t in = 0; in < n; in++) {
            p[in + n * n] = start[in];
        }
        y[n] = loss_fun(start.data(), n);
        ++icount;

        omp_pfor__(j, 0, n,
                   auto x = start[j];
                           start[j] = start[j] + step[j] * del;
                           for (size_t i = 0; i < n; i++) {
                               p[i + j * n] = start[i];
                           }
                           y[j] = loss_fun(start.data(), n);
                           start[j] = x;
        )
        icount += n - 1;
        //
        //  The simplex construction is complete.
        //
        //  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
        //  the vertex of the simplex to be replaced.
        //
        double ylo = y[0];
        int ilo = 0;

        for (size_t i = 1; i < nn; i++) {
            if (y[i] < ylo) {
                ylo = y[i];
                ilo = i;
            }
        }
        //
        //  Inner loop.
        //
        while (true) {
            if (nm_parameters.max_iteration_number_ <= icount) {
                break;
            }
            ynewlo = y[0];
            ihi = 0;

            for (i = 1; i < nn; i++) {
                if (ynewlo < y[i]) {
                    ynewlo = y[i];
                    ihi = i;
                }
            }
            //
            //  Calculate PBAR, the centroid of the simplex vertices
            //  excepting the vertex with Y value YNEWLO.
            //
            omp_pfor_i__ (0, n,
                          auto z = 0.0;
                                  for (j = 0; j < nn; j++) {
                                      z = z + p[i + j * n];
                                  }
                                  z = z - p[i + ihi * n];
                                  pbar[i] = z / dn;
            )
            //
            //  Reflection through the centroid.
            //
            for (size_t i = 0; i < n; i++) {
                pstar[i] = pbar[i] + rcoeff * (pbar[i] - p[i + ihi * n]);
            }
            ystar = loss_fun(pstar.data(), n);
            ++icount;
            //
            //  Successful reflection, so extension.
            //
            if (ystar < ylo) {
                for (size_t i = 0; i < n; i++) {
                    p2star[i] = pbar[i] + ecoeff * (pstar[i] - pbar[i]);
                }
                y2star = loss_fun(p2star.data(), n);
                ++icount;
                //
                //  Check extension.
                //
                if (ystar < y2star) {
                    for (size_t i = 0; i < n; i++) {
                        p[i + ihi * n] = pstar[i];
                    }
                    y[ihi] = ystar;
                }
                    //
                    //  Retain extension or contraction.
                    //
                else {
                    for (size_t i = 0; i < n; i++) {
                        p[i + ihi * n] = p2star[i];
                    }
                    y[ihi] = y2star;
                }
            }
                //
                //  No extension.
                //
            else {
                ell = 0;
                for (i = 0; i < nn; i++) {
                    if (ystar < y[i]) {
                        ell = ell + 1;
                    }
                }

                if (1 < ell) {
                    for (size_t i = 0; i < n; i++) {
                        p[i + ihi * n] = pstar[i];
                    }
                    y[ihi] = ystar;
                }
                    //
                    //  Contraction on the Y(IHI) side of the centroid.
                    //
                else if (ell == 0) {
                    omp_pfor_i__ (0, n, p2star[i] = pbar[i] + ccoeff * (p[i + ihi * n] - pbar[i]))
                    y2star = loss_fun(p2star.data(), n);
                    ++icount;
                    //
                    //  Contract the whole simplex.
                    //
                    if (y[ihi] < y2star) {
                        omp_pfor__(j, 0, nn,
                                   for (size_t i = 0; i < n; i++) {
                                       p[i + j * n] = (p[i + j * n] + p[i + ilo * n]) * 0.5;
                                       xmin[i] = p[i + j * n];
                                   }
                                           y[j] = loss_fun(xmin.data(), n);
                        )
                        icount += nn - 1;
                        ylo = y[0];
                        ilo = 0;

                        for (i = 1; i < nn; i++) {
                            if (y[i] < ylo) {
                                ylo = y[i];
                                ilo = i;
                            }
                        }
                        continue;
                    }
                        //
                        //  Retain contraction.
                        //
                    else {
                        for (size_t i = 0; i < n; i++) {
                            p[i + ihi * n] = p2star[i];
                        }
                        y[ihi] = y2star;
                    }
                }
                    //
                    //  Contraction on the reflection side of the centroid.
                    //
                else if (ell == 1) {
                    for (size_t i = 0; i < n; i++) {
                        p2star[i] = pbar[i] + ccoeff * (pstar[i] - pbar[i]);
                    }
                    y2star = loss_fun(p2star.data(), n);
                    ++icount;
                    //
                    //  Retain reflection?
                    //
                    if (y2star <= ystar) {
                        for (size_t i = 0; i < n; i++) {
                            p[i + ihi * n] = p2star[i];
                        }
                        y[ihi] = y2star;
                    } else {
                        for (size_t i = 0; i < n; i++) {
                            p[i + ihi * n] = pstar[i];
                        }
                        y[ihi] = ystar;
                    }
                }
            }
            //
            //  Check if YLO improved.
            //
            if (y[ihi] < ylo) {
                ylo = y[ihi];
                ilo = ihi;
            }
            --jcount;

            if (0 < jcount) continue;

            //
            //  Check to see if minimum reached.
            //
            if (icount <= nm_parameters.max_iteration_number_) {
                jcount = check_for_convergence;

                z = 0;
                for (i = 0; i < nn; i++) z += y[i];
                x = z / dnn;

                z = 0;
                for (i = 0; i < nn; i++) z += std::pow(y[i] - x, 2);

                if (z <= rq) {
                    break;
                }
            }
        }
        //
        //  Factorial tests to check that YNEWLO is a local minimum.
        //
        omp_pfor_i__ (0, n, xmin[i] = p[i + ilo * n])
        ynewlo = y[ilo];

        if (nm_parameters.max_iteration_number_ < icount)
            break;

        ifault = 0;

        std::mutex mux;
        std::atomic<size_t> red_icount(0);
        omp_pfor_i__ (0, n,
                      auto del = step[i] * eps;
                              xmin[i] = xmin[i] + del;
                              auto z = loss_fun(xmin.data(), n);
                              red_icount += 1;
                              std::scoped_lock l(mux);
                              if (z < ynewlo) {
                                  ifault = 2;
                              } else {
                                  xmin[i] = xmin[i] - del - del;
                                  z = loss_fun(xmin.data(), n);
                                  red_icount += 1;
                                  if (z < ynewlo) {
                                      ifault = 2;
                                  } else {
                                      xmin[i] = xmin[i] + del;
                                  }
                              }
        )
        icount += red_icount;

        if (ifault == 0) break;

        //
        //  Restart the procedure.
        //
        start = xmin;
        memcpy(start.data(), xmin.data(), n * sizeof(start[0]));
        del = eps;
    }

    LOG4_INFO("NM: number of iterations: " << icount);

    return std::make_pair(ynewlo, xmin);
}

}
}