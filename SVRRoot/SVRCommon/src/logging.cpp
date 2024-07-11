//
// Created by zarko on 5/26/24.
//

#include <ipp/ippcore.h>
#include "common/logging.hpp"

class load_unload
{
public:
    load_unload()
    {
        // ip_errchk(ippInit());
    }

    ~load_unload()
    {
    }
};

const auto l = []() {
    return load_unload();
}();

std::string cufft_get_error_string(const cufftResult s)
{
    switch (s) {
        case CUFFT_SUCCESS:
            return "Any CUFFT operation is successful.";
        case CUFFT_INVALID_PLAN:
            return "CUFFT is passed an invalid plan handle.";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT failed to allocate GPU memory.";
        case CUFFT_INVALID_TYPE:
            return "The user requests an unsupported type.";
        case CUFFT_INVALID_VALUE:
            return "The user specifies a bad memory pointer.";
        case CUFFT_INTERNAL_ERROR:
            return "Used for all internal driver errors.";
        case CUFFT_EXEC_FAILED:
            return "CUFFT failed to execute an FFT on the GPU.";
        case CUFFT_SETUP_FAILED:
            return "The CUFFT library failed to initialize.";
        case CUFFT_INVALID_SIZE:
            return "The user specifies an unsupported FFT size.";
        default:
            return "Unknown error";
    }
}
