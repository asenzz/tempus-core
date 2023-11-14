#pragma once

#include "OnlineSMOSVR.hpp"

namespace svr {

template<typename S>
bool OnlineSVR::load_online_svr(OnlineSVR &osvr, S &input_stream)
{
    osvr.clear();

    try {
        // Title
        std::string trash;
        double X;
        int X2;
        int samples_dimension;
        int lag_count;
        double C, epsilon, kp1, kp2;
        std::string kernel_type;

        input_stream >> trash >> osvr.samples_trained_number;
        input_stream >> trash >> samples_dimension;
        input_stream >> trash >> C;
        osvr.svr_parameters.set_svr_C(C);
        input_stream >> trash >> epsilon;
        osvr.svr_parameters.set_svr_epsilon(epsilon);
        input_stream >> trash >> kernel_type;
        osvr.svr_parameters.set_kernel_type(get_kernel_type_from_string(kernel_type));
        input_stream >> trash >> kp1;
        osvr.svr_parameters.set_svr_kernel_param(kp1);
        input_stream >> trash >> kp2;
        osvr.svr_parameters.set_svr_kernel_param2(kp2);
        input_stream >> trash >> lag_count;
        osvr.svr_parameters.set_lag_count(lag_count);
        input_stream >> trash >> X2;
        osvr.stabilized_learning = X2 > 0;
        input_stream >> trash >> X2;
        osvr.save_kernel_matrix = X2 > 0;

        if (trash != "SaveKernelMatrix:")
            throw std::runtime_error("Error during loading.");

        int xrows, xcols;
        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;

        int i;
        for (i = 0; i < xrows; i++) {
            vektor<double> *Sample = new vektor<double>(xrows);
            for (int j = 0; j < xcols; j++) {
                input_stream >> X;
                Sample->add(X);
            }
            osvr.X->add_row_ref(Sample);
        }

        input_stream >> trash >> xrows;
        if (trash != "Y:")
            throw std::runtime_error("Error during loading labels.");

        for (i = 0; i < xrows; i++) {
            input_stream >> X;
            osvr.Y->add(X);
        }

        int tmp;

        input_stream >> trash >> tmp;
        for (i = 0; i < tmp; i++) {
            input_stream >> X2;
            osvr.p_support_set_indexes->add(X2);
        }

        input_stream >> trash >> tmp;
        for (i = 0; i < tmp; i++) {
            input_stream >> X2;
            osvr.p_error_set_indexes->add(X2);
        }

        input_stream >> trash >> tmp;
        for (i = 0; i < tmp; i++) {
            input_stream >> X2;
            osvr.p_remaining_set_indexes->add(X2);
        }
        if (trash != "RemainingSet:")
            throw std::runtime_error("Error during loading remaining set.");

        input_stream >> trash >> tmp;
        for (i = 0; i < tmp; i++) {
            input_stream >> X;
            osvr.p_weights->add(X);
        }

        input_stream >> trash >> osvr.bias;
        input_stream >> trash >> xrows;
        input_stream >> trash >> xcols;

        for (i = 0; i < xrows; i++) {
            vektor<double> *sample = new vektor<double>(xrows);
            for (int j = 0; j < xcols; j++) {
                input_stream >> X;
                sample->add(X);
            }
            osvr.p_r_matrix->add_row_ref(sample);
        }

        if (trash != "RmCols:")
            throw std::runtime_error("Error during loading 3.");

        if (osvr.save_kernel_matrix)
            osvr.build_kernel_matrix();
    }
    catch (const std::exception &ex) {
        LOG4_ERROR("Error. Data corrupted. " << ex.what());
        osvr.clear();
        return false;
    }

    return true;
}


template<typename S>
bool OnlineSVR::save_online_svr(const OnlineSVR &osvr, S &output_stream)
{
    output_stream.precision(std::numeric_limits<double>::max_digits10);

    try {
        output_stream << "SamplesTrainedNumber: " << osvr.samples_trained_number << std::endl;
        output_stream << "SamplesDimension: " << osvr.X->get_length_cols() << std::endl;
        output_stream << "C: " << osvr.svr_parameters.get_svr_C() << std::endl;
        output_stream << "Epsilon: " << osvr.svr_parameters.get_svr_epsilon() << std::endl;
        output_stream << "KernelType: " << kernel_type_to_string(osvr.svr_parameters.get_kernel_type()) << std::endl;
        output_stream << "KernelParam: " << osvr.svr_parameters.get_svr_kernel_param() << std::endl;
        output_stream << "KernelParam2: " << osvr.svr_parameters.get_svr_kernel_param2() << std::endl;
        output_stream << "LagCount: " << osvr.svr_parameters.get_lag_count() << std::endl;
        output_stream << "StabilizedLearning: " << (osvr.stabilized_learning ? 1 : 0) << std::endl;
        output_stream << "SaveKernelMatrix: " << (osvr.save_kernel_matrix ? 1 : 0) << std::endl;

        output_stream << "X.rows: " << osvr.X->get_length_rows() << std::endl;
        output_stream << "X.cols: " << osvr.X->get_length_cols() << std::endl;

        int i;
        for (i = 0; i < osvr.X->get_length_rows(); i++)
            for (int j = 0; j < osvr.X->get_length_cols(); j++)
                output_stream << osvr.X->get_value(i, j) << " ";

        output_stream << std::endl << std::endl << "Y: " << osvr.Y->size() << std::endl;

        for (i = 0; i < osvr.Y->size(); i++)
            output_stream << osvr.Y->get_value(i) << " ";

        output_stream << std::endl << std::endl << "SupportSet: " << osvr.get_support_set_elements_number()
                      << std::endl;
        for (i = 0; i < osvr.get_support_set_elements_number(); i++)
            output_stream << osvr.p_support_set_indexes->get_value(i) << std::endl;

        output_stream << std::endl << "ErrorSet: " << osvr.get_error_set_elements_number() << std::endl;
        for (i = 0; i < osvr.get_error_set_elements_number(); i++)
            output_stream << osvr.p_error_set_indexes->get_value(i) << std::endl;

        output_stream << std::endl << "RemainingSet: " << osvr.get_remaining_set_elements_number() << std::endl;
        for (i = 0; i < osvr.get_remaining_set_elements_number(); i++)
            output_stream << osvr.p_remaining_set_indexes->get_value(i) << std::endl;

        output_stream << std::endl << std::endl << std::endl << "Weights: " << osvr.p_weights->size()
                      << std::endl;
        for (i = 0; i < osvr.p_weights->size(); i++)
            output_stream << osvr.p_weights->get_value(i) << "\n";

        output_stream << std::endl << "Bias: " << osvr.bias << std::endl;

        output_stream << std::endl << "RmRows: " << osvr.p_r_matrix->get_length_rows() << std::endl;
        output_stream << "RmCols: " << osvr.p_r_matrix->get_length_cols() << std::endl;
        for (i = 0; i < osvr.p_r_matrix->get_length_rows(); i++)
            for (int j = 0; j < osvr.p_r_matrix->get_length_cols(); j++)
                output_stream << osvr.p_r_matrix->get_value(i, j) << "\n";
    } catch (...) {
        LOG4_ERROR("Error. It's impossible to complete the save.");
    }
    return true;
}

}
