#pragma once

#include "onlinesvr.hpp"
#include "common/parallelism.hpp"

namespace svr {

datamodel::kernel_type_e get_kernel_type_from_string(const std::string &kernel_type_str);

std::string kernel_type_to_string(const datamodel::kernel_type_e kernel_type);

template<typename S> bool
OnlineMIMOSVR::save_online_svr(const OnlineMIMOSVR &osvr, S &output_stream)
{
    output_stream.precision(std::numeric_limits<double>::max_digits10);

    const auto svr_parameters = osvr.get_svr_parameters();
    try {
        output_stream << "SamplesDimension: " << osvr.p_features->n_cols << std::endl;
        output_stream << "C: " << svr_parameters.get_svr_C() << std::endl;
        output_stream << "Epsilon: " << svr_parameters.get_svr_epsilon() << std::endl;
        output_stream << "KernelType: " << kernel_type_to_string(svr_parameters.get_kernel_type()) << std::endl;
        output_stream << "KernelParam: " << svr_parameters.get_svr_kernel_param() << std::endl;
        output_stream << "KernelParam2: " << svr_parameters.get_svr_kernel_param2() << std::endl;
        output_stream << "LagCount: " << svr_parameters.get_lag_count() << std::endl;
        output_stream << "MimoType: " << int(osvr.get_mimo_type()) << std::endl;
        output_stream << "MultistepLen: " << osvr.get_multistep_len() << std::endl;
        output_stream << "SaveKernelMatrix: " << (osvr.save_kernel_matrix ? 1 : 0) << std::endl;

        output_stream << "X.rows: " << osvr.p_features->n_rows << std::endl;
        output_stream << "X.cols: " << osvr.p_features->n_cols << std::endl;

        for (size_t i = 0; i < osvr.p_features->n_rows; i++)
            for (size_t j = 0; j < osvr.p_features->n_cols; j++)
                output_stream << osvr.p_features->at(i, j) << " ";

        output_stream << std::endl;

        output_stream << "Y.rows: " << osvr.p_labels->n_rows << std::endl;
        output_stream << "Y.cols: " << osvr.p_labels->n_cols << std::endl;

        for (size_t i = 0; i < osvr.p_labels->n_rows; i++)
            for (size_t j = 0; j < osvr.p_labels->n_cols; j++)
                output_stream << osvr.p_labels->at(i, j) << " ";

        output_stream << std::endl;

        for (auto &kv: osvr.main_components) {
            for (size_t i = 0; i < kv.second.chunk_weights.size(); ++i) {
                arma::mat weights = kv.second.chunk_weights[i];
                output_stream << "Weights.rows: " << weights.n_rows << std::endl;
                output_stream << "Weights.cols: " << weights.n_cols << std::endl;
                for (size_t r = 0; r < weights.n_rows; r++)
                    for (size_t c = 0; c < weights.n_cols; c++)
                        output_stream << weights(r, c) << " ";
                output_stream << std::endl;
            }
        }

    } catch (const std::exception &ex) {
        LOG4_ERROR("It's impossible to complete the save." << ex.what());
    }
    return true;
}

template<typename S>
OnlineMIMOSVR_ptr
OnlineMIMOSVR::load_online_svr(S &input_stream)
{
    OnlineMIMOSVR_ptr p_loaded_object;

    try {
        // Title
        std::string trash;
        int X2;
        int samples_dimension;
        int lag_count;
        double C, epsilon, kp1, kp2;
        std::string kernel_type;
        size_t multistep_len;
        size_t i_mimo_type;
        MimoType mimo_type;
        bool save_kernel_matrix;
        datamodel::SVRParameters_ptr svr_parameters = std::make_shared<datamodel::SVRParameters>();

        input_stream >> trash >> samples_dimension;
        input_stream >> trash >> C;
        svr_parameters->set_svr_C(C);
        input_stream >> trash >> epsilon;
        svr_parameters->set_svr_epsilon(epsilon);
        input_stream >> trash >> kernel_type;
        svr_parameters->set_kernel_type(get_kernel_type_from_string(kernel_type));
        input_stream >> trash >> kp1;
        svr_parameters->set_svr_kernel_param(kp1);
        input_stream >> trash >> kp2;
        svr_parameters->set_svr_kernel_param2(kp2);
        input_stream >> trash >> lag_count;
        svr_parameters->set_lag_count(lag_count);
        input_stream >> trash >> i_mimo_type;
        mimo_type = i_mimo_type == 1 ? MimoType::single : MimoType::twin;
        input_stream >> trash >> multistep_len;
        input_stream >> trash >> X2;
        save_kernel_matrix = X2 > 0;

        if (trash != "SaveKernelMatrix:") LOG4_THROW("Error during loading.");

        p_loaded_object = std::make_shared<OnlineMIMOSVR>(svr_parameters, mimo_type, multistep_len);

        size_t xrows, xcols;
        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;

        if (trash != "X.cols:") LOG4_THROW("Error during loading labels.");

        p_loaded_object->save_kernel_matrix = save_kernel_matrix;
        p_loaded_object->p_features->resize(xrows, xcols);
        for (size_t i = 0; i < xrows; i++)
            for (size_t j = 0; j < xcols; j++)
                input_stream >> p_loaded_object->p_features->at(i, j);

        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;

        if (trash != "Y.cols:") LOG4_THROW("Error during loading labels.");

        p_loaded_object->p_labels->resize(xrows, xcols);
        for (size_t i = 0; i < xrows; i++)
            for (size_t j = 0; j < xcols; j++)
                input_stream >> p_loaded_object->p_labels->at(i, j);

        for (auto &kv: p_loaded_object->main_components) {
            for (size_t i = 0; i < kv.second.chunk_weights.size(); ++i) {
                arma::mat weights = kv.second.chunk_weights[i];
                input_stream >> trash >> xrows;
                input_stream >> trash >> xcols;

                if (trash != "Weights.cols:") LOG4_THROW("Error during loading labels.");

                kv.second.chunk_weights[i].resize(xrows, xcols);
                for (size_t r = 0; r < xrows; ++r)
                    for (size_t c = 0; c < xcols; ++c)
                        input_stream >> kv.second.chunk_weights[i](r, c);
            }
        }

        p_loaded_object->ixs = p_loaded_object->get_indexes(xrows, *p_loaded_object->p_svr_parameters);
        if (p_loaded_object->save_kernel_matrix) {
            __omp_tpfor_i(size_t, 0, p_loaded_object->ixs.size(),
                          init_kernel_matrix(
                                  *p_loaded_object->p_svr_parameters,
                                  p_loaded_object->p_features->rows(p_loaded_object->ixs[i]),
                                  p_loaded_object->p_labels->rows(p_loaded_object->ixs[i]),
                                  p_loaded_object->p_kernel_matrices->at(i),
                                  p_loaded_object->p_manifold)
            )
        }
    } catch (const std::exception &ex) {
        LOG4_ERROR("Data corrupted. " << ex.what());
        throw;
    }

    return p_loaded_object;
}

template<typename S>
bool OnlineMIMOSVR::save_onlinemimosvr_no_weights_no_kernel(const OnlineMIMOSVR &osvr, S &output_stream)
{
    output_stream.precision(std::numeric_limits<double>::max_digits10);

    const auto svr_parameters = osvr.get_svr_parameters();
    try {
        output_stream << "C: " << svr_parameters.get_svr_C() << std::endl;
        output_stream << "Epsilon: " << svr_parameters.get_svr_epsilon() << std::endl;
        output_stream << "KernelType: " << kernel_type_to_string(svr_parameters.get_kernel_type()) << std::endl;
        output_stream << "KernelParam: " << svr_parameters.get_svr_kernel_param() << std::endl;
        output_stream << "KernelParam2: " << svr_parameters.get_svr_kernel_param2() << std::endl;
        output_stream << "LagCount: " << svr_parameters.get_lag_count() << std::endl;
        output_stream << "MimoType: " << int(osvr.get_mimo_type()) << std::endl;
        output_stream << "MultistepLen: " << osvr.get_multistep_len() << std::endl;

        output_stream << "X.rows: " << osvr.p_features->n_rows << std::endl;
        output_stream << "X.cols: " << osvr.p_features->n_cols << std::endl;

        for (size_t i = 0; i < osvr.p_features->n_rows; i++)
            for (size_t j = 0; j < osvr.p_features->n_cols; j++)
                output_stream << osvr.p_features->at(i, j) << " ";

        output_stream << std::endl;

        output_stream << "Y.rows: " << osvr.p_labels->n_rows << std::endl;
        output_stream << "Y.cols: " << osvr.p_labels->n_cols << std::endl;

        for (size_t i = 0; i < osvr.p_labels->n_rows; i++)
            for (size_t j = 0; j < osvr.p_labels->n_cols; j++)
                output_stream << osvr.p_labels->at(i, j) << " ";

    } catch (const std::exception &ex) {
        LOG4_ERROR("It's impossible to complete the save." << ex.what());
    }
    return true;
}

template<typename S>
OnlineMIMOSVR_ptr
OnlineMIMOSVR::load_onlinemimosvr_no_weights_no_kernel(S &input_stream)
{
    OnlineMIMOSVR_ptr p_loaded_object;

    try {
        // Title
        std::string trash;
        int lag_count;
        double _C, _epsilon, _kp1, _kp2;
        std::string _kernel_type;
        size_t _multistep_len;
        size_t _i_mimo_type;
        MimoType _mimo_type;
        datamodel::SVRParameters_ptr _svr_parameters = std::make_shared<datamodel::SVRParameters>();

        input_stream >> trash >> _C;
        _svr_parameters->set_svr_C(_C);
        input_stream >> trash >> _epsilon;
        _svr_parameters->set_svr_epsilon(_epsilon);
        input_stream >> trash >> _kernel_type;
        _svr_parameters->set_kernel_type(get_kernel_type_from_string(_kernel_type));
        input_stream >> trash >> _kp1;
        _svr_parameters->set_svr_kernel_param(_kp1);
        input_stream >> trash >> _kp2;
        _svr_parameters->set_svr_kernel_param2(_kp2);
        input_stream >> trash >> lag_count;
        _svr_parameters->set_lag_count(lag_count);
        input_stream >> trash >> _i_mimo_type;
        _mimo_type = (_i_mimo_type == 1) ? MimoType::single : MimoType::twin;
        input_stream >> trash >> _multistep_len;

        if (trash != "MultistepLen:") LOG4_THROW("Error during loading.");

        p_loaded_object = std::make_shared<OnlineMIMOSVR>(_svr_parameters, _mimo_type, _multistep_len);

        size_t xrows, xcols;
        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;


        if (trash != "X.cols:") LOG4_THROW("Error during loading labels.");

        p_loaded_object->p_features->resize(xrows, xcols);
        for (size_t i = 0; i < xrows; i++)
            for (size_t j = 0; j < xcols; j++)
                input_stream >> p_loaded_object->p_features->at(i, j);

        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;

        if (trash != "Y.cols:")
            LOG4_THROW("Error during loading labels.");

        p_loaded_object->p_labels->resize(xrows, xcols);
        for (size_t i = 0; i < xrows; i++)
            for (size_t j = 0; j < xcols; j++)
                input_stream >> p_loaded_object->p_labels->at(i, j);

    } catch (const std::exception &ex) {
        LOG4_ERROR("Data corrupted. " << ex.what());
        throw;
    }

    return p_loaded_object;
}

}
