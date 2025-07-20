//
// Created by zarko on 16/06/2025.
//

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <iostream>
#include <fstream>
#include <LightGBM/c_api.h>
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/logging.hpp"
#include "kernel_gbm.hpp"
#include "appcontext.hpp"
#include "../../SVRBusiness/include/ScalingFactorService.hpp"

namespace svr {
namespace kernel {

#define LGBM_MAXBIN "255"

std::string get_lgbm_dataset_parameters()
{
    return "max_bin=" LGBM_MAXBIN " use_missing=false save_binary=true";
}

std::string get_lgbm_core_parameters(const uint16_t gpu_id)
{
    std::stringstream s;
    s << "objective=regression tree_learner=data seed=123 learning_rate=" << PROPS.get_k_learn_rate() << " num_iterations=" << PROPS.get_k_epochs()
        << " early_stopping_round=200 metric=l2 num_leaves=256 feature_fraction=0.7 bagging_fraction=0.7 bagging_freq=25 force_col_wise=true device_type=gpu num_threads=" <<
            C_n_cpu << " " << get_lgbm_dataset_parameters() << " gpu_device_id=" << gpu_id; //  min_data_in_leaf=200
#ifndef NDEBUG
    s << " verbosity=2 ";
#endif
    /* device_type=cuda is buggy and crashes LightGBM, test and enable the following line when LGBM is fixed
    s << "deterministic=true lambda_l1=0.1 lambda_l2=1.0 gpu_device_id=" << gpu_id << " device_type=cuda num_gpu=" << common::gpu_handler_1::get().get_gpu_devices_count();
    */
#ifdef INSTANCE_WEIGHTS
    s << "weight_column=0"
#endif
    return s.str();
}

#define T double

template<> kernel_gbm<T>::kernel_gbm(datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

template<> arma::Mat<T> kernel_gbm<T>::kernel(const arma::Mat<T> &X /* predict features */, const arma::Mat<T> &Xy /* trained features */) const
{
    LOG4_BEGIN();
    const int64_t n_samples = X.n_cols * Xy.n_cols;
    const int64_t n_manifold_features = X.n_rows + Xy.n_rows;
    BoosterHandle booster;
    int num_iterations;
    {
        std::stringstream s(parameters.get_tft_model());
        std::stringstream decompressed_stream;
        boost::iostreams::filtering_istream in;
        in.push(boost::iostreams::bzip2_decompressor());
        in.push(s);
        decompressed_stream << in.rdbuf();
        decompressed_stream.flush();
        lg_errchk(LGBM_BoosterLoadModelFromString(decompressed_stream.str().data(), &num_iterations, &booster))
    }
    arma::fmat manifold_features_t(n_manifold_features, n_samples, ARMA_DEFAULT_FILL);
    OMP_FOR_(n_samples, collapse(2) SSIMD)
    for (uint32_t i = 0; i < X.n_cols; ++i)
        for (uint32_t j = 0; j < Xy.n_cols; ++j)
            manifold_features_t.col(i + X.n_cols * j) = arma::conv_to<arma::fvec>::from(arma::join_cols(X.col(i), Xy.col(j)));

    arma::mat res(X.n_cols, Xy.n_cols, ARMA_DEFAULT_FILL);
    int64_t out_len;
    common::gpu_context ctx;
    const auto gbm_parameters = get_lgbm_core_parameters(ctx.phy_id());
    lg_errchk(LGBM_BoosterPredictForMat(booster, manifold_features_t.mem, C_API_DTYPE_FLOAT32, n_samples, n_manifold_features, 1, C_API_PREDICT_NORMAL, 0, 0, gbm_parameters.c_str(), &out_len, res.memptr()));
    LOG4_TRACE("Predicted " << out_len << ", labels " << common::present(res));
    LGBM_BoosterFree(booster);
    assert(parameters.get_svr_kernel_param() != 0);
    common::unscale_I(res, parameters.get_svr_kernel_param(), parameters.get_min_Z());
    return res;
}

void lgbm_log(const char *const lgbm_message)
{
    LOG4_DEBUG(lgbm_message);
}

template<> void kernel_gbm<T>::init(const arma::Mat<T> &X_t, const arma::Mat<T> &Y)
{
    LOG4_BEGIN();
    assert(Y.n_cols == 1);
    const uint32_t n_samples = X_t.n_cols;
    const uint32_t n_samples_2 = n_samples * n_samples;
    const uint32_t n_manifold_features = X_t.n_rows * 2;
    // LightGBM uses row-major matrices
    arma::fmat manifold_features_t(n_manifold_features, n_samples_2, ARMA_DEFAULT_FILL);
    arma::fmat manifold_labels(n_samples_2, Y.n_cols, ARMA_DEFAULT_FILL);
    OMP_FOR_(n_samples_2, collapse(2) firstprivate(n_samples_2) SSIMD)
    for (DTYPE(n_samples) i = 0; i < n_samples; ++i)
        for (DTYPE(n_samples) j = 0; j < n_samples; ++j) {
            const auto row = i + n_samples * j;
            manifold_labels.row(row) = arma::conv_to<arma::frowvec>::from(Y.row(i) - Y.row(j));
            manifold_features_t.col(row) = arma::conv_to<arma::fvec>::from(arma::join_cols(X_t.col(i), X_t.col(j)));
        }
    double sf, dc;
    business::ScalingFactorService::scale_calc_I(manifold_labels, sf, dc, common::C_input_obseg_labels);
    LOG4_TRACE("Scaled manifold labels " << common::present(manifold_labels) << ", scaling factor " << sf << ", dc offset " << dc);
    parameters.set_svr_kernel_param(sf);
    parameters.set_min_Z(dc);

    lg_errchk(LGBM_SetMaxThreads(C_n_cpu));
    DatasetHandle train_dataset;
    {
        const auto gbm_dataset_parameters = get_lgbm_dataset_parameters();
        lg_errchk(LGBM_DatasetCreateFromMat(manifold_features_t.mem, C_API_DTYPE_FLOAT32,n_samples_2, n_manifold_features, 1, // is_row_major = 1 (row-major order)
            gbm_dataset_parameters.c_str(), nullptr, &train_dataset));
    }

    lg_errchk(LGBM_DatasetSetField(train_dataset, "label", manifold_labels.mem, n_samples_2, C_API_DTYPE_FLOAT32));

    lg_errchk(LGBM_RegisterLogCallback(lgbm_log));

    BoosterHandle booster;
    common::gpu_context ctx;
    const auto gbm_parameters = get_lgbm_core_parameters(ctx.phy_id());
    lg_errchk(LGBM_BoosterCreate(train_dataset, gbm_parameters.c_str(), &booster));
    int train_finished = 0;
    auto iter = PROPS.get_k_epochs() + 1;
    assert(iter);
    while (train_finished != 1 && --iter) lg_errchk(LGBM_BoosterUpdateOneIter(booster, &train_finished));

    int64_t model_size = 0;
    LGBM_BoosterSaveModelToString(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, 0, &model_size, nullptr);
    std::vector<char> model_str(model_size);
    lg_errchk(LGBM_BoosterSaveModelToString(booster, 0, 0, C_API_FEATURE_IMPORTANCE_SPLIT, model_size, &model_size, model_str.data()));
    lg_errchk(LGBM_BoosterFree(booster));
    lg_errchk(LGBM_DatasetFree(train_dataset));

    std::ostringstream compressed_stream;
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::bzip2_compressor());
    out.push(compressed_stream);
    out.write(model_str.data(), model_size);
    boost::iostreams::close(out);
    compressed_stream.flush();
    parameters.set_tft_model(compressed_stream.str());
    LOG4_DEBUG("Saved LighGBM model with size " << model_size << " bytes to parameters " << parameters);
    LOG4_END();
}

template<> arma::Mat<T> kernel_gbm<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_THROW("Not implemeneted.");
    return {};
}

template<> void kernel_gbm<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    LOG4_THROW("Not implemented.");
}

template<> void kernel_gbm<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    LOG4_THROW("Not implemented.");
}
} // kernel
} // svr
