//
// Created by zarko on 19/03/2025.
//

#ifndef SVR_KERNEL_BASE_TPP
#define SVR_KERNEL_BASE_TPP

#include "appcontext.hpp"
#include "kernel_base.hpp"
#include "kernel_base.cuh"
#include "common/compatibility.hpp"
#include "calc_cache.hpp"
#include "pprune.hpp"
#include "SVRParametersService.hpp"
#include "model/SVRParameters.hpp"
#include "util/math_utils.hpp"

namespace svr {
namespace kernel {
template<typename T> arma::Mat<T> get_reference_Z(const arma::Mat<T> &y)
{
    const uint32_t n = y.n_rows;
    const arma::Mat<T> y_t = y.t();
    arma::Mat<T> r(n, n, ARMA_DEFAULT_FILL);
    OMP_FOR_i(n) r.row(i) = y(i, 0) - y_t;
    LOG4_TRACE("Prepared reference kernel matrix " << common::present(r) << " from labels " << common::present(y));
    return r;
}

template<typename T> arma::Mat<T> get_reference_Z(const arma::Mat<T> &y1, const arma::Mat<T> &y2)
{
    const uint32_t m = y1.n_rows;
    const uint32_t n = y2.n_rows;
    const arma::Mat<T> y2_t = y2.t();
    arma::Mat<T> r(m, n, ARMA_DEFAULT_FILL);
    OMP_FOR_i(m) r.row(i) = y1(i, 0) - y2_t;
    LOG4_TRACE("Prepared reference kernel matrix " << common::present(r) << " from labels " << common::present(y1) << " and " << common::present(y2));
    return r;
}

template<typename T> datamodel::SVRParameters &kernel_base<T>::get_parameters()
{
    return parameters;
}

template<typename T> datamodel::SVRParameters kernel_base<T>::get_parameters() const
{
    return parameters;
}

template<typename T> kernel_base<T>::kernel_base(datamodel::SVRParameters &p) : parameters(p)
{
}

template<typename T> kernel_base<T>::~kernel_base() = default;

template<typename T> void kernel_base<T>::update(datamodel::OnlineSVR &model, const uint32_t chunk_ix, const arma::Mat<T> &x, const arma::Mat<T> &y)
{
    LOG4_TRACE("Ignoring update kernel " << parameters.get_kernel_type() << " with X " << common::present(x) << " and Y " << common::present(y));
}
template<typename T> void kernel_base<T>::d_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const
{
    d_distances(d_X, d_X, m, n, n, d_Z, custream);
}

template<typename T> void kernel_base<T>::d_kernel_from_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const
{
    d_kernel_from_distances(d_X, d_Z, m, n, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2(), custream);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(const arma::Mat<T> &X) const
{
    return kernel(X, X);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(const arma::Mat<T> &X) const
{
    return distances(X, X);
}

template<typename T> void kernel_base<T>::kernel_from_distances_I(arma::Mat<T> &Kz) const
{
    kernel::kernel_from_distances(Kz.memptr(), Kz.n_rows, Kz.n_cols, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2());
    LOG4_TRACE("Prepared K " << common::present(Kz) << " with parameters " << parameters);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel_from_distances(const arma::Mat<T> &Z) const
{
    arma::Mat<T> K(arma::size(Z), ARMA_DEFAULT_FILL);
    kernel::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2());
    LOG4_TRACE("Prepared K " << common::present(K) << " with parameters " << parameters << ", from Z " << common::present(Z));
    return K;
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const
{
    LOG4_BEGIN();
    return cc.get_Ky(*this, X, X, X_time, X_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(
    business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const
{
    return cc.get_Ky(*this, X, Xy, X_time, Xy_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const
{
    return cc.get_Zy(*this, X, X, X_time, X_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(
    business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const
{
    return cc.get_Zy(*this, X, Xy, X_time, Xy_time);
}

template<typename T> void kernel_base<T>::wrapup(datamodel::OnlineSVR &model, const uint32_t chunk_ix) const
{
    LOG4_BEGIN();

    if (parameters.get_svr_kernel_param() == 0) parameters.set_svr_kernel_param(1);

    model.set_params(parameters, chunk_ix);
    LOG4_INFO("Tuned final parameters " << parameters);

    if (model.get_id()) {
        if (APP.svr_parameters_service.exists(parameters))
            APP.svr_parameters_service.remove(parameters);
        APP.svr_parameters_service.save(parameters);
    }

    LOG4_END();
}

template<typename T> void kernel_base<T>::init(datamodel::OnlineSVR &model, const uint32_t chunk_ix)
{
    constexpr uint8_t D = 1;
    // static const auto equiexp = std::log(std::sqrt(PROPS.get_tune_max_lambda())) / M_LN2;
    static const auto bounds1 = [] {
        arma::mat r(4, 2, ARMA_DEFAULT_FILL);
        r.col(0).zeros();
        r.col(1).fill(PROPS.get_tune_max_fback());
        r(0, 1) = PROPS.get_tune_max_tau();
        return r;
    }();
    static const auto bounds2 = [] {
        arma::mat r(D, 2, ARMA_DEFAULT_FILL);
        r(0, 0) = 0;
        r(0, 1) = PROPS.get_tune_max_lambda();
        return r;
    }();

    tbb::mutex chunk_preds_l;
    auto best_score = std::numeric_limits<double>::max();
    cutuner cv(model.get_X(chunk_ix), model.get_Y(chunk_ix), parameters);
    auto costF = [&](const double x[], double *const f) {
        const auto [score, gamma, min] = cv.phase1(x[0], x[1], x[2], x[3]);
        *f = score;
        const tbb::mutex::scoped_lock lk(chunk_preds_l);
        if (score < best_score) {
            parameters.set_svr_kernel_param(gamma);
            parameters.set_kernel_param3(*x);
            parameters.set_H_feedback(x[1]);
            parameters.set_D_feedback(x[2]);
            parameters.set_V_feedback(x[3]);
            parameters.set_min_Z(min);
            LOG4_TRACE("New best score distances " << score << ", previous best " << best_score << ", improvement " << common::imprv(score, best_score) << "pc, parameters " <<
                parameters << ", opt arg " << common::to_string(x, 4));
            best_score = score;
        }
    };
    (void) optimizer::pprune(optimizer::pprune::C_default_algo, PROPS.get_tune_particles1(), bounds1, costF, PROPS.get_tune_iteration1(), 0, 0, {}, {},
                             std::min<uint32_t>(PROPS.get_tune_iteration1(), PROPS.get_opt_depth()));
    cv.prepare_second_phase(parameters);
    auto costF2 = [&](const double x[], double *const f) {
        const auto [score, gamma, min] = cv.phase2(*x);
        *f = score;
        const tbb::mutex::scoped_lock lk(chunk_preds_l);
        if (score < best_score) {
            parameters.set_svr_kernel_param(gamma);
            parameters.set_svr_kernel_param2(*x);
            parameters.set_min_Z(min);
            LOG4_TRACE("New best score kernel " << score << ", previous best " << best_score << ", improvement " << common::imprv(score, best_score) << "pc, parameters " <<
                parameters << ", opt arg " << *x);
            best_score = score;
        }
    };
    (void) optimizer::pprune(optimizer::pprune::C_default_algo, PROPS.get_tune_particles2(), bounds2, costF2, PROPS.get_tune_iteration2(), 0, 0, {}, {},
                             std::min<uint32_t>(PROPS.get_tune_iteration2(), PROPS.get_opt_depth()));

    assert(parameters.get_svr_kernel_param() != 0);
    wrapup(model, chunk_ix);
}
}
}

#endif //SVR_KERNEL_BASE_TPP
