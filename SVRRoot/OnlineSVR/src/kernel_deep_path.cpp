//
// Created by zarko on 19/03/2025.
//

#include <oneapi/mkl/types.hpp>
#include <armadillo>
#include "kernel_deep_path.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace kernel {

// #define RANDOMIZE_MANIFOLD_INDEXES // TODO Test, seems like it worsens results

// Specializations
#define T double

template<> kernel_deep_path<T>::kernel_deep_path(datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

template<> kernel_deep_path<T>::kernel_deep_path(kernel_base<T> &k) : kernel_base<T>(k.get_parameters())
{
}


/* L diff matrix format:
 * L0 - L0, L0 - L1, L0 - L2, ..., L0 - Lm
 * L1 - L0, L1 - L1, L1 - L2, ..., L1 - Lm
 * ...
 * Ln - L0, Ln - L1, Ln - L2, ..., Ln - Lm
 */

template<> void kernel_deep_path<T>::init(const uint32_t parent_projection, datamodel::Dataset_ptr &p_dataset, const arma::mat &features_t, const arma::mat &labels, const bpt::ptime &last_time)
{
    if (parameters.get_manifold()) LOG4_WARN("Manifold already initialized!");

    const auto n_samples = features_t.n_cols;
    const auto n_samples_2 = n_samples * n_samples;
    const auto n_manifold_features = features_t.n_rows * 2;

    auto p_manifold_parameters = otr(parameters);
    switch (parameters.get_kernel_type()) {
        case datamodel::e_kernel_type::DEEP_PATH:
            p_manifold_parameters->set_kernel_type(datamodel::e_kernel_type::PATH);
            break;
        default:
            LOG4_THROW("Kernel type " << parameters.get_kernel_type() << " not handled.");
    }
    p_manifold_parameters->set_chunk_index(0);
    p_manifold_parameters->set_svr_kernel_param(0);
    p_manifold_parameters->set_svr_decremental_distance(n_samples_2);
    {
        auto &fm = p_manifold_parameters->get_feature_mechanics();
        fm.stretches = arma::fvec(n_manifold_features, arma::fill::ones);
        fm.shifts = arma::u32_vec(n_manifold_features, arma::fill::zeros);
    }
    auto p_manifold = otr<datamodel::OnlineSVR>(0, 0, datamodel::t_param_set{p_manifold_parameters}, p_dataset);
    p_manifold->set_projection(parent_projection + 1);
    const auto p_manifold_features = ptr<arma::mat>(n_samples_2, n_manifold_features, ARMA_DEFAULT_FILL);
    const auto p_manifold_labels = ptr<arma::mat>(n_samples_2, labels.n_cols, ARMA_DEFAULT_FILL);
    const auto p_manifold_lastknowns = ptr<arma::vec>(n_samples_2, arma::fill::zeros);
    const auto p_manifold_weights = ptr<arma::vec>(n_samples_2, arma::fill::ones); // TODO Implement
    OMP_FOR_(n_samples_2, collapse(2) firstprivate(n_samples) SSIMD)
    for (DTYPE(n_samples) i = 0; i < n_samples; ++i)
        for (DTYPE(n_samples) j = 0; j < n_samples; ++j) {
            const auto row = i + n_samples * j;
            p_manifold_labels->row(row) = labels.row(i) - labels.row(j);
            p_manifold_features->row(row) = arma::join_cols(features_t.col(i), features_t.col(j)).t();
        }

    LOG4_DEBUG("Generated " << common::present(*p_manifold_labels) << " manifold label matrix and " << common::present(*p_manifold_features) <<
        " manifold feature matrix from " << common::present(features_t) << " feature matrix and " << common::present(labels) << " label matrix.");
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, p_manifold_weights, last_time);
    assert(p_manifold_features->has_nonfinite() == false && p_manifold_labels->has_nonfinite() == false);
    assert(p_manifold_lastknowns->has_nonfinite() == false && p_manifold_weights->has_nonfinite() == false);
    parameters.set_manifold(p_manifold);
    LOG4_END();
}


template<> arma::Mat<T> kernel_deep_path<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    auto p_manifold = parameters.get_manifold();
    assert(p_manifold);
    // X and Xy are transposed therefore the acrobatics below
    arma::mat Ky(X.n_cols, Xy.n_cols);
    OMP_FOR(X.n_cols)
    for (uint32_t i = 0; i < X.n_cols; ++i) Ky.row(i) = p_manifold->predict(arma::join_cols(common::extrude_cols(X.col(i), Xy.n_cols), Xy).t()).t();
    LOG4_TRACE("Prepared kernel " << common::present(Ky) << " with parameters " << parameters << ", from X " << common::present(X) << " and Xy " << common::present(Xy));
    return Ky;
}

template<> arma::Mat<T> kernel_deep_path<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_THROW("This kernel does not implement distances.");
    return {};
}


template<> void kernel_deep_path<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    LOG4_THROW("This kernel does not implement kernel from distances.");
}

template<> void kernel_deep_path<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    LOG4_THROW("This kernel does not implement distances.");
}

#if 0
arma::mat OnlineMIMOSVR::manifold_predict(arma::mat x_predict_t, const boost::posix_time::ptime &time) const
{
    assert(ixs.size() == 1 && train_feature_chunks_t.size() == 1 && train_label_chunks.size() == 1);

    constexpr uint16_t chunk_ix = 0;
    const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
    business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, get_params(chunk_ix).get_lag_count(), chunk_sf, x_predict_t);
    arma::mat result(x_predict_t.n_cols, p_labels->n_cols);
    const arma::mat &features_t = train_feature_chunks_t.front();
    const auto manifold_interleave = PROPS.get_interleave();
    const arma::SizeMat predict_size(x_predict_t.n_rows + features_t.n_rows, CDIVI(features_t.n_cols, manifold_interleave));
    arma::uvec predict_ixs(predict_size.n_cols);
#ifdef RANDOMIZE_MANIFOLD_INDEXES
    common::threadsafe_uniform_int_distribution<uint32_t> rand_int(0, manifold_interleave - 1);
    // auto gen = common::reproducibly_seeded_64<xso::rng64>(); // TODO Freezes on operator() fix
    auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(manifold_interleave));
    tbb_pfor_i__(0, predict_size.n_cols,
        const auto rand_int_g = rand_int(gen);
        predict_ixs[i] = common::bounce<uint32_t>(i * manifold_interleave + rand_int_g, features_t.n_cols - 1);
    )
#else
    for (uint32_t i = 0; i < predict_size.n_cols; ++i) predict_ixs[i] = i * manifold_interleave;
#endif
    const auto predict_labels = train_label_chunks.front().rows(predict_ixs);
    const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
#pragma omp parallel ADJ_THREADS(std::max<int32_t>(1, common::gpu_handler<4>::get().get_max_gpu_threads() - int32_t(p_manifold->get_num_chunks())))
#pragma omp single
    {
        OMP_TASKLOOP_(x_predict_t.n_cols,)
        for (uint32_t i_p = 0; i_p < x_predict_t.n_cols; ++i_p) {
            arma::mat pred_features_t(predict_size, ARMA_DEFAULT_FILL);
            OMP_TASKLOOP_(pred_features_t.n_cols, firstprivate(i_p))
            for (uint32_t i_r = 0; i_r < pred_features_t.n_cols; ++i_r)
                pred_features_t.col(i_r) = arma::join_cols(x_predict_t.col(i_p), features_t.col(predict_ixs[i_r]));
            result[i_p] = arma::mean(arma::vectorise(predict_labels + p_manifold->predict(pred_features_t, bpt::not_a_date_time)));
            business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, result[i_p]);
        }
    }
    return result;
}
#endif

}
}
