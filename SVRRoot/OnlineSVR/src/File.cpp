
#include <iostream>
#include <fstream>
#include "onlinesvr_persist.hpp"


namespace svr {

// I/O Operations
OnlineMIMOSVR_ptr OnlineMIMOSVR::load_online_mimosvr(const char *filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file) {
        LOG4_ERROR("Error. File not found." << filename);
        throw std::runtime_error("Error. File not found.");
    }

    return load_online_svr(file);
}

bool OnlineMIMOSVR::save_online_mimosvr(const char *filename)
{
    bool res;
    // Open the file
    std::ofstream file(filename, std::ios::out);
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

    OnlineMIMOSVR_ptr OnlineMIMOSVR::load_onlinemimosvr_no_weights_no_kernel(const char *filename)
    {
        std::ifstream file(filename, std::ios::in);
        if (!file) {
            LOG4_ERROR("Error. File not found." << filename);
            throw std::runtime_error("Error. File not found.");
        }

        return load_onlinemimosvr_no_weights_no_kernel(file);
    }

    bool OnlineMIMOSVR::save_onlinemimosvr_no_weights_no_kernel(const char *filename)
    {
        bool res;
        // Open the file
        std::ofstream file(filename, std::ios::out);
        if (!file) {
            LOG4_ERROR("Error. It's impossible to create the file.");
            return false;
        }

        res = save_onlinemimosvr_no_weights_no_kernel(*this, file);
        if (res == false) LOG4_ERROR("Failed saving model to " << filename);

        // Close the file
        file.close();

        return res;
    }



} // svr
