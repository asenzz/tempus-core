//
// Created by zarko on 11/11/21.
//

#include <armadillo>
#include <vector>
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


namespace svr {

static std::map<double /* sum */, arma::mat /* labels */> cached_labels;

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


std::tuple<matrix_ptr, matrix_ptr>
prepare_manifold_train_data(
        const arma::mat &F,
        const arma::mat &L,
        double &scaling_factor)
{
#ifdef INTERLACE_MANIFOLD_FACTOR
    const auto n_samples = F.n_rows * F.n_rows / INTERLACE_MANIFOLD_FACTOR;
#else
    const auto n_samples = F.n_rows * F.n_rows;
#endif
    auto p_manifold_features = std::make_shared<arma::mat>(n_samples, F.n_cols * 2);
    auto p_manifold_labels = std::make_shared<arma::mat>(n_samples, L.n_cols);

    arma::cube L_diff(L.n_rows, L.n_rows, L.n_cols);
    const arma::rowvec L_t = L.t();

#pragma omp parallel for collapse(2)
    for (size_t r = 0; r < L.n_rows; ++r)
        for (size_t c = 0; c < L.n_cols; ++c)
            L_diff.slice(c).row(r) = L(r, c) - L_t.row(c); // Asymmetric distance matrix

    p_manifold_features->set_size(L.n_rows * L.n_rows, L.n_cols);
#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < L.n_rows; ++i)
        for (size_t j = 0; j < L.n_rows; ++j)
            for (size_t k = 0; k < L.n_cols; ++k) {
                const auto r = i + j * L.n_rows; // TODO Double check if i * L.n_rows + j works better for row index
#ifdef INTERLACE_MANIFOLD_FACTOR
                if (r % INTERLACE_MANIFOLD_FACTOR) continue;
#endif
                p_manifold_labels->at(r, k) = L_diff(i, j, k);
                p_manifold_features->row(r) = arma::join_rows(F.row(i), F.row(j));
            }

    LOG4_DEBUG("Generated " << arma::size(*p_manifold_labels) << " manifold label matrix and " << arma::size(*p_manifold_features) <<
                            " manifold feature matrix from " << arma::size(F) << " feature matrix and " << arma::size(L) << " label matrix.");
    scaling_factor = arma::median(arma::vectorise(arma::abs(*p_manifold_labels))) / common::C_input_obseg_labels;
    *p_manifold_labels /= scaling_factor;
    return {p_manifold_features, p_manifold_labels};
}


void
OnlineMIMOSVR::init_manifold(const datamodel::SVRParameters_ptr &p)
{
    if (p_manifold) {
        LOG4_ERROR("Manifold already initialized!");
        return;
    }

    datamodel::SVRParameters_ptr p_manifold_parameters = std::make_shared<datamodel::SVRParameters>(*p);
    switch (p->get_kernel_type()) {
        case datamodel::kernel_type_e::DEEP_PATH:
            p_manifold_parameters->set_kernel_type(datamodel::kernel_type_e::PATH);
            break;
        case datamodel::kernel_type_e::DEEP_PATH2:
            p_manifold_parameters->set_kernel_type(datamodel::kernel_type_e::DEEP_PATH);
            break;
        default:
            LOG4_ERROR("Kernel type " << int(p->get_kernel_type()) << " not handled.");
            return;
    }

    const auto param_set = std::make_shared<datamodel::t_param_set>();
    param_set->emplace(p_manifold_parameters);
    p_manifold = std::make_shared<OnlineMIMOSVR>(param_set, multistep_len, chunk_size);
    const auto [p_manifold_features, p_manifold_labels] = prepare_manifold_train_data(*p_features, *p_labels, p_manifold->labels_scaling_factor);
    p_manifold->batch_train(p_manifold_features, p_manifold_labels);
}


datamodel::SVRParameters_ptr OnlineMIMOSVR::is_manifold() const
{
    for (const auto &p: *p_param_set)
        if (p->is_manifold())
            return p;
    return nullptr;
}


arma::mat OnlineMIMOSVR::manifold_predict(const arma::mat &x_predict) const
{
    arma::mat predict_features(x_predict.n_rows * p_features->n_rows, p_features->n_cols + x_predict.n_cols);
    arma::mat base_labels(x_predict.n_rows * p_labels->n_rows, p_labels->n_cols);
#pragma omp parallel for collapse(2)
    for (size_t i_p = 0; i_p < x_predict.n_rows; ++i_p)
        for (size_t i_x = 0; i_x < p_features->n_rows; ++i_x) {
            const auto r = i_p + i_x * x_predict.n_rows;
            predict_features.row(r) = arma::join_rows(p_features->row(i_x), x_predict.row(i_p));
            base_labels.row(r) = p_labels->row(i_x);
        }
    auto result = p_manifold->predict(predict_features);
    result += base_labels;
    return result;
}

} // namespace svr {
