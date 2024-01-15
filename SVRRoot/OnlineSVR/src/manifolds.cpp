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

arma::mat /* desired kernel function distances Z matrix of size n_labels x n_labels, K = 1 - Z / (2 * g^2) */
OnlineMIMOSVR::get_reference_distance_matrix(const arma::mat &L)
{
    LOG4_BEGIN();

#ifdef EMO_LIN_KERNEL    // Emo lin kernel
    return L * L.t();
#else
    // My way
    // Use for training a manifold
    const size_t n_labels = L.n_rows;
    arma::mat labels_abs_diff(n_labels, n_labels);
    const arma::rowvec row_L = L.t();

    __tbb_pfor(row_ix, 0, n_labels, labels_abs_diff.row(row_ix) = L(row_ix, 0) - row_L) // Asymmetric distance matrix
    LOG4_DEBUG("Generated reference distances matrix " << common::present(labels_abs_diff) << " for labels " << common::present(L));

    return labels_abs_diff;
#endif
}

std::tuple<std::vector<arma::mat>, std::vector<double>>
OnlineMIMOSVR::get_reference_matrices(const arma::mat &L, const datamodel::SVRParameters &svr_parameters)
{
    const auto indexes = get_indexes(L.n_rows, svr_parameters);
    const auto num_chunks = indexes.size();

    std::vector<arma::mat> res(num_chunks);
    std::vector<double> norm_res(num_chunks);
    __tbb_pfor_i(0, num_chunks,
        res[i] = get_reference_distance_matrix(L.rows(indexes[i]));
        norm_res[i] = arma::norm(res[i], "fro");
        LOG4_DEBUG("Reference matrix " << i << " of size " << arma::size(res[i]) << " for labels " << arma::size(L) << ", norm " << norm_res[i]);
    )
    return {res, norm_res};
}

void generate_manifold_labels(const arma::mat &labels, arma::mat &manifold_labels)
{
    const auto ideal_kmatrix = OnlineMIMOSVR::get_reference_distance_matrix(labels);
    manifold_labels.set_size(OnlineMIMOSVR::get_manifold_nrows(labels.n_rows), 1);
    size_t output_ix = 0;
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j <= i; ++j)
            manifold_labels(output_ix++, 0) = ideal_kmatrix(i, j);
    }

    if (manifold_labels.n_rows != output_ix)
        LOG4_THROW("manifold_labels.n_rows " << manifold_labels.n_rows << " != " << output_ix << " output_ix");
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
    std::vector<matlab::data::MATLABFieldIdentifier> field_names;
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
    LOG4_DEBUG("Manifold label rows " << output_ix << ", manifold labels size " << arma::size(manifold_labels) << ", labels size " << arma::size(labels));
}


arma::mat
manifold_join_feature_rows(const arma::mat &lhs, const arma::mat &rhs)
{
#ifdef PSD_MANIFOLD
    if (lhs.empty() || rhs.empty() || arma::size(lhs) != arma::size(rhs))
        LOG4_WARN("Feature vector sizes " << arma::size(lhs) << " and " << arma::size(rhs) << " check failed!");
    const auto max_cols = std::min<size_t>(lhs.n_cols, rhs.n_cols);
    for (size_t c = 0; c < max_cols; ++c) {
        if (lhs(0, c) < rhs(0, c))
            return arma::join_rows(lhs, rhs);
        else if (lhs(0, c) > rhs(0, c))
            return arma::join_rows(rhs, lhs);
    }
    LOG4_DEBUG("Feature vectors identical in " << max_cols << " columns, lhs: " << lhs << ", rhs: " << rhs);
#endif
    return arma::join_rows(lhs, rhs);
}


static void
prepare_manifold_train_data(
        const arma::mat &x_train,
        const arma::mat &y_train,
        matrix_ptr &p_manifold_features,
        matrix_ptr &p_manifold_labels,
        double &scaling_factor)
{
    const auto n_rows = x_train.n_rows;
#ifdef FAST_MANIFOLD_TRAIN
    const auto fast_n_rows = OnlineMIMOSVR::get_manifold_nrows(n_rows);
    if (!p_manifold_features)
        p_manifold_features = std::make_shared<arma::mat>(fast_n_rows, x_train.n_cols * 2);
    else
        p_manifold_features->set_size(fast_n_rows, x_train.n_cols * 2);
    if (!p_manifold_labels) p_manifold_labels = std::make_shared<arma::mat>(fast_n_rows, y_train.n_cols);
    generate_manifold_labels(y_train, *p_manifold_labels);
    if (p_manifold_labels->n_rows != fast_n_rows)
        LOG4_THROW("Incorrect size of manifold labels " << arma::size(*p_manifold_labels) << ", expected " << fast_n_rows);
    size_t row_ix = 0;
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j <= i; ++j)
            p_manifold_features->row(row_ix++) = manifold_join_feature_rows(x_train.row(i), x_train.row(j));
    }
#else
    if (!p_manifold_features)
        p_manifold_features = std::make_shared<arma::mat>(n_rows * n_rows, x_train.n_cols * 2);
    else
        p_manifold_features->set_size(n_rows * n_rows, x_train.n_cols * 2);
    if (!p_manifold_labels)
        p_manifold_labels = std::make_shared<arma::mat>(n_rows * n_rows, y_train.n_cols);
    else
        p_manifold_labels->set_size(n_rows * n_rows, y_train.n_cols);
    __tbb_pfor_i(size_t, 0, n_rows,
        for (size_t j = 0; j < n_rows; ++j) {
            const auto row_ix = i + j * n_rows; // TODO Correct?
            p_manifold_labels->row(row_ix) = arma::abs(y_train.row(i) - y_train.row(j));
            p_manifold_features->row(row_ix) = arma::join_rows(x_train.row(i), x_train.row(j));
        }
    )
#endif
    if (row_ix != p_manifold_features->n_rows) LOG4_THROW("row_ix " << row_ix << " != " << p_manifold_features->n_rows << " p_manifold_features->n_rows");
    LOG4_DEBUG("Generated " << arma::size(*p_manifold_labels) << " manifold label matrix and " << arma::size(*p_manifold_features) <<
                         " manifold feature matrix from " << arma::size(x_train) << " feature matrix and " << arma::size(y_train) << " label matrix, feature rows " << row_ix);
    scaling_factor = arma::median(arma::vectorise(*p_manifold_labels)) / common::C_input_obseg_labels;
    *p_manifold_labels /= scaling_factor;
}


OnlineMIMOSVR_ptr
OnlineMIMOSVR::init_manifold(const arma::mat &features, const arma::mat &labels, const datamodel::SVRParameters &svr_parameters)
{
    datamodel::SVRParameters_ptr p_manifold_parameters = std::make_shared<datamodel::SVRParameters>(svr_parameters);
    if (svr_parameters.get_kernel_type() == datamodel::kernel_type_e::DEEP_PATH)
        p_manifold_parameters->set_kernel_type(datamodel::kernel_type_e::PATH);
    else if (svr_parameters.get_kernel_type() == datamodel::kernel_type_e::DEEP_PATH2)
        p_manifold_parameters->set_kernel_type(datamodel::kernel_type_e::DEEP_PATH);
    else
        return nullptr;

    auto p_manifold = std::make_shared<OnlineMIMOSVR>(p_manifold_parameters, MimoType::single, 1);
    matrix_ptr p_manifold_features, p_manifold_labels;
    prepare_manifold_train_data(features, labels, p_manifold_features, p_manifold_labels, p_manifold->labels_scaling_factor);
#ifdef PSEUDO_ONLINE
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, false, nullptr, true);
#else
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, false, nullptr, false);
#endif
    return p_manifold;
}


void OnlineMIMOSVR::init_manifold(const arma::mat &features, const arma::mat &labels)
{
    if (p_svr_parameters->get_kernel_type() != datamodel::kernel_type_e::DEEP_PATH and
        p_svr_parameters->get_kernel_type() != datamodel::kernel_type_e::DEEP_PATH2)
        return;
    p_manifold = init_manifold(features, labels, *p_svr_parameters);
}


size_t OnlineMIMOSVR::get_manifold_nrows(const size_t n_rows)
{
#ifdef PSD_MANIFOLD
    return n_rows * n_rows / 2. + n_rows / 2.;
#else
    return n_rows * n_rows;
#endif
}


arma::mat prepare_manifold_predict_data(const arma::mat &x_train, const arma::mat &x_predict)
{
    arma::mat result(x_predict.n_rows * x_train.n_rows, x_train.n_cols + x_predict.n_cols);
    size_t out_row_ix = 0;
    for (size_t i_p = 0; i_p < x_predict.n_rows; ++i_p)
        for (size_t i_x = 0; i_x < x_train.n_rows; ++i_x)
            result.row(out_row_ix++) = manifold_join_feature_rows(x_train.row(i_x), x_predict.row(i_p));
    return result;
}


arma::mat OnlineMIMOSVR::init_predict_manifold_matrix(
        const arma::mat &x_train,
        const arma::mat &x_predict,
        const OnlineMIMOSVR_ptr &p_manifold)
{
    auto result = p_manifold->chunk_predict(prepare_manifold_predict_data(x_train, x_predict));
    result.reshape(x_predict.n_rows, x_train.n_rows); // TODO Verify!
    return result;
}

} // namespace svr {
