#include <iostream>
#include <fstream>
#include "OnlineSMOSVR.tcc"

using namespace std;

namespace svr {

// I/O Operations
bool OnlineSVR::load_online_svr(const char *filename)
{
    bool res;
    ifstream file(filename, ios::in);
    if (!file) {
        LOG4_ERROR("Error. File not found." << filename);
        return false;
    }

    res = load_online_svr(*this, file); //Evgeniy's edition
    if (res == false) LOG4_ERROR("Failed loading model from " << filename);

    file.close();
    return res;
}

void OnlineSVR::load_online_svr(std::stringstream &input_stream) {
    if (load_online_svr(*this, input_stream) == false)
        throw std::logic_error("Failed loading model from stream.");
}


bool OnlineSVR::save_onlinesvr(const char *filename) const
{
    bool res;
    // Open the file
    ofstream file(filename, ios::out);
    if (!file) {
        LOG4_ERROR("Error. It's impossible to create the file.");
        return false;
    }

    res = save_online_svr(*this, file);
    if (res == false) LOG4_ERROR("Failed saving model to " << filename);

    // Close the file
    file.close();

    return res;
}


void OnlineSVR::save_online_svr(std::stringstream &output_stream) const
{
    OnlineSVR::save_online_svr(*this, output_stream);
}


void OnlineSVR::load_online_svr(
        const svm_model &libsvm_model,
        const vmatrix<double> &learning_data,
        const vektor<double> &reference_data)
{
    if (libsvm_model.param.svm_type != EPSILON_SVR)
        throw std::logic_error("trying to load a libsvm model of incorrect type");
    samples_trained_number = libsvm_model.l;
    //read X and Y sets
    const int *sv_indices = libsvm_model.sv_indices;

    for (int i = 0; i < samples_trained_number; i++) {
        int current_sv_index = sv_indices[i];
        Y->add(reference_data.get_value(current_sv_index - 1));
        X->add_row_copy(learning_data.get_row_ptr(current_sv_index - 1));
        p_support_set_indexes->add(i);
    }
    //read weights
    for (int i = 0; i < samples_trained_number; i++) p_weights->add(1.);
    
    //read bias
    bias = -libsvm_model.rho[0]; //it's negative

    // Kernel vmatrix
    if (save_kernel_matrix) build_kernel_matrix();
}

} // svr
