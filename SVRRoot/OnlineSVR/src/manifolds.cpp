//
// Created by zarko on 11/11/21.
//

#include <armadillo>
#include <cmath>
#include <cstddef>
#include <random>
#include <random>
#include <vector>
#include <xoshiro.h>
#include "common/compatibility.hpp"
#include "appcontext.hpp"

#ifdef EXPERIMENTAL_FEATURES
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#pragma GCC diagnostic pop
#endif


#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "SVRParametersService.hpp"
#include "DQScalingFactorService.hpp"


namespace svr {
namespace datamodel {


#if 0 // Matlab code to generate ideal kernel matrix prototype
std::scoped_lock lg (mlab_mx);
    LOG4_BEGIN();

    const auto labels_sum = arma::sum(arma::vectorise(labels));
    const auto cached_labels_iter = cached_labels.find(labels_sum);
    if (cached_labels_iter != cached_labels.end()) {
        LOG4_DEBUG("Returning cached labels for sum " << labels_sum);
        manifold_labels = cached_labels_iter->second;
        return;
    }

    if (labels.n_cols != 1) LOG4_THROW("Labels dimensions are incorrect " << arma::size(labels));

    std::string cmd = "syms ";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < i; ++j) {
            cmd += "k" + std::to_string(i) + "_" + std::to_string(j) + " ";
        }
    }
    cmd += ";\n";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "l" + std::to_string(i) + "=" + common::to_string_with_precision(labels[i]) + ";";
    }
    cmd += "M=[";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < labels.n_elem; ++j) {
            if (i == j)
                cmd += "1";
            else if (i < j)
                cmd += "k" + std::to_string(j) +  "_" + std::to_string(i);
            else
                cmd += "k" + std::to_string(i) +  "_" + std::to_string(j);
            if (j < labels.n_elem - 1)
                cmd += ",";
        }
        if (i < labels.n_elem - 1)
            cmd += ";";
    }
    cmd += "];\nW = [";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "1";
        if (i < labels.n_elem - 1) cmd += ";";
    }
    cmd += "];\nL=[";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "l" + std::to_string(i);
        if (i < labels.n_elem - 1) cmd += ";";
    }
    cmd += "];\nres=solve(M*W-L,";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < i; ++j) {
            cmd += "k" + std::to_string(i) + "_" + std::to_string(j);
            if (i == labels.n_elem - 1 and j == labels.n_elem - 2) continue;
            cmd += ",";
        }
    }
    cmd += ");";
    //LOG4_DEBUG("Matlab routine is:\n"+cmd);

    static std::unique_ptr<matlab::engine::MATLABEngine> p_matlab = matlab::engine::startMATLAB();
    static matlab::data::ArrayFactory factory;

    std::u16string u16cmd;
    common::from_utf8(cmd, u16cmd);
    LOG4_FILE("/tmp/manifold_cmd.txt", ""<<cmd);
    p_matlab->eval(u16cmd);
    matlab::data::StructArray res = p_matlab->getVariable(u"res");
    const matlab::data::ArrayDimensions dims = res.getDimensions();
    LOG4_DEBUG("Result dimensions " << dims[0] << "x" << dims[1]);

    // Get number of fields
    const size_t numFields = res.getNumberOfFields();
    LOG4_DEBUG("Result has " << numFields << " fields");
    if (numFields != (labels.n_elem * labels.n_elem) / 2 - labels.n_elem / 2)
        LOG4_THROW("Wrong number of fields returned " << numFields);
/*
    // Get the struct array fieldnames
    auto fields = res.getFieldNames();
    std::deque<matlab::data::MATLABFieldIdentifier> field_names;
    for (const auto& field_name : fields) { field_names.push_back(field_name);
        LOG4_DEBUG("Result got field " << std::string(name));
 */
    size_t output_ix = 0;
    manifold_labels.set_size(labels.n_elem * labels.n_elem / 2 + labels.n_elem / 2, 1);
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                manifold_labels(output_ix++, 0) = 1;
                continue;
            }
            const std::string field_name = "k" + std::to_string(i) + "_" + std::to_string(j);
            std::u16string u16str;
            common::from_utf8(field_name, u16str);
            p_matlab->eval(u"tmp = double(res." + u16str + u");");
            matlab::data::TypedArray<double> field = p_matlab->getVariable(u"tmp");
            //LOG4_TRACE("Field " << field_name << " type " << int(field.getType()) << ", size " << field.getDimensions()[0] << "x" << field.getDimensions()[1] << ", elements " << field.getNumberOfElements());
            if (field.getDimensions()[0] < 1 or field.getDimensions()[1] < 1)
                LOG4_THROW("Empty symbol returned for " << field_name);
            manifold_labels(output_ix++, 0) = field[0][0];
            /*
                    matlab::data::ArrayDimensions data_dims = field.getDimensions();
                    for (size_t j=0; j<data_dims[0]; ++j){
                        for (size_t k=0; k<data_dims[1]; ++k){
                            std::cout << j << ", " << k << ": " << field[j][k] << " ";
                        }
                        std::cout << std::endl;
                    }
            */
        }
    }
    p_matlab->evalAsync(u"clear;");
    cached_labels[labels_sum] = manifold_labels;
    LOG4_END();
#endif /* EXPERIMENTAL_FEATURES */

void
OnlineMIMOSVR::init_manifold(const datamodel::SVRParameters_ptr &p, const bpt::ptime &last_value_time)
{
    if (p_manifold) {
        LOG4_ERROR("Manifold already initialized!");
        return;
    }

    auto p_manifold_parameters = otr(*p);
    switch (p->get_kernel_type()) {
        case datamodel::e_kernel_type::DEEP_PATH:
            p_manifold_parameters->set_kernel_type(datamodel::e_kernel_type::PATH);
            break;
        default:
            LOG4_ERROR("Kernel type " << p->get_kernel_type() << " not handled.");
            return;
    }
    assert(ixs.size() == 1);
    const arma::mat prepared_features = feature_chunk_t(ixs[0]).t(); // TODO Remove dual transpose
    const uint32_t n_rows = prepared_features.n_rows;
    const auto n_learning_rows = std::min<uint32_t>(n_rows, p->get_svr_decremental_distance());
    const uint32_t n_manifold_samples = n_rows * n_rows / PROPS.get_interleave();
    p_manifold_parameters->set_svr_decremental_distance(n_learning_rows * n_learning_rows / PROPS.get_interleave());
    p_manifold_parameters->set_chunk_index(0);
    {
        auto &fm = p_manifold_parameters->get_feature_mechanics();
        fm.stretches = arma::fvec(prepared_features.n_cols * 2, arma::fill::ones); // arma::join_cols(fm.stretches, fm.stretches);
        fm.shifts = arma::u32_vec(prepared_features.n_cols * 2, arma::fill::zeros); // arma::join_cols(fm.shifts, fm.shifts);
        // fm.skips = arma::join_cols(fm.skips, fm.skips);
    }
    p_manifold = otr<OnlineMIMOSVR>(0, model_id, t_param_set{p_manifold_parameters}, p_dataset);
    p_manifold->projection += 1;

    auto p_manifold_features = ptr<arma::mat>(n_manifold_samples, prepared_features.n_cols * 2, arma::fill::none);
    auto p_manifold_labels = ptr<arma::mat>(p_manifold_features->n_rows, p_labels->n_cols, arma::fill::none);
    auto p_manifold_lastknowns = ptr<arma::vec>(p_manifold_features->n_rows, arma::fill::none);
    auto p_manifold_weights = ptr<arma::mat>(arma::size(*p_manifold_labels), arma::fill::none);

    /* L diff matrix:
     * L0 - L0, L0 - L1, L0 - L2, ..., L0 - Lm
     * L1 - L0, L1 - L1, L1 - L2, ..., L1 - Lm
     * ...
     * Ln - L0, Ln - L1, Ln - L2, ..., Ln - Lm
     */
    std::uniform_int_distribution<uint32_t> rand_int(0, PROPS.get_interleave() - 1);
    // auto gen = common::reproducibly_seeded_64<xso::rng64>(); // TODO Freezes on operator() fix
    // auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(PROPS.get_interleave()));
    t_omp_lock r_lock;
    const auto manifold_interleave = PROPS.get_interleave();
    OMP_FOR_(n_rows * n_rows * p_labels->n_cols, SSIMD collapse(3))
    for (uint32_t i = 0; i < n_rows; ++i)
        for (uint32_t j = 0; j < n_rows; ++j)
            for (uint32_t k = 0; k < p_labels->n_cols; ++k) {
                const auto r = i + j * n_rows; // Column by column ordering
                if (r % manifold_interleave == 0) {
                    const auto r_out = r / manifold_interleave;
                    if (r_out < n_manifold_samples) {
                        /*
                        r_lock.set();
                        const auto rand_int_g = rand_int(gen);
                        r_lock.unset();
                        */
                        const auto rand_int_g = 0;
                        const auto r_i = common::bounce<uint32_t>(i + rand_int_g, n_rows - 1);
                        const auto r_j = common::bounce<uint32_t>(j + rand_int_g, n_rows - 1);
                        MANIFOLD_SET(
                                p_manifold_features->row(r_out), p_manifold_labels->at(r_out, k),
                                prepared_features.row(r_j), p_labels->at(r_j, k),
                                prepared_features.row(r_i), p_labels->at(r_i, k))
                    }
                }
            }

    LOG4_DEBUG("Generated " << arma::size(*p_manifold_labels) << " manifold label matrix and " << arma::size(*p_manifold_features) <<
                            " manifold feature matrix from " << arma::size(prepared_features) << " feature matrix and " << arma::size(*p_labels) << " label matrix.");
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, p_manifold_lastknowns, p_manifold_weights, last_value_time);
}

datamodel::SVRParameters_ptr OnlineMIMOSVR::is_manifold() const
{
    return business::SVRParametersService::is_manifold(param_set);
}

arma::mat OnlineMIMOSVR::manifold_predict(const arma::mat &x_predict, const boost::posix_time::ptime &time) const
{
    arma::mat result(x_predict.n_rows, p_labels->n_cols);
    const auto manifold_interleave = PROPS.get_interleave();
    const arma::SizeMat predict_size(p_features->n_rows / manifold_interleave, x_predict.n_cols + p_features->n_cols);
    arma::uvec predict_rows(predict_size.n_rows);
    OMP_TASKLOOP(p_features->n_rows)
    for (uint32_t i_x = 0; i_x < p_features->n_rows; ++i_x)
        if (i_x % manifold_interleave)
            predict_rows[i_x / manifold_interleave] = i_x;

    const auto predict_labels = p_labels->rows(predict_rows);

#pragma omp parallel ADJ_THREADS(x_predict.n_rows * p_features->n_rows)
#pragma omp single
    {
        OMP_TASKLOOP(x_predict.n_rows)
        for (uint32_t i_p = 0; i_p < x_predict.n_rows; ++i_p) {
            common::memory_manager::get().barrier();
            result.row(i_p) =
                    arma::mean(arma::vectorise(predict_labels + p_manifold->predict(
                            arma::join_rows(common::extrude_cols<double>(x_predict.row(i_p), predict_size.n_rows), p_features->rows(predict_rows)),
                            bpt::not_a_date_time)));
        }
    }
    return result;
}

OnlineMIMOSVR_ptr OnlineMIMOSVR::get_manifold()
{
    return p_manifold;
}

} // datamodel
} // namespace svr
