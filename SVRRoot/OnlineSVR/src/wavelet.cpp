#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <execution>
#include "common/constants.hpp"
#include "common/logging.hpp"
#include "util/math_utils.hpp"
#include "spectral_transform.hpp"
#include "wavelet_coefs.hpp"


// #define WAVELET_TRANSFORM_DEBUGGING
#ifdef WAVELET_TRANSFORM_DEBUGGING
#include <iostream>
#endif

namespace svr {

spectral_transform::wavelet::wavelet(std::string name) : name_(name)
{
// TODO: make sure to propagate the correct wavelet order instead of extracting it from the name;
// or remove the parameter.
    int extracted_order = 0;
    {
        int digit = 1;
        int digits = 0;
        std::string::iterator it = name.end() - 1;
        while (it != name.end() && *it >= '0' && *it <= '9') {
            ++digits;
            extracted_order += digit * int(*it - '0');
            digit *= 10;
            --it;
        }
        name.resize(name.size() - digits);
    }
    if (extracted_order != 0) {
        order_ = extracted_order;
    } else if (name == std::string("haar")) {
        order_ = 1;
    } else {
        LOG4_THROW("wavelet_transform::wavelet::wavelet(): could not extract order of wavelet");
    }

    constexpr double sqrt2_double = 1.4142135623730951454746218587388284504413604736328125;

    std::transform(C_default_exec_policy, name.begin(), name.end(), name.begin(), ::tolower);
    name_ = name;

    if (name == std::string("haar")) {

        /* the same as db1 */
        auto wave = wavelet("db1");
        *this = wave;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;
        name_ = name;
        return;

        /* Reverse biorthogonal wavelets family */
    } else if (name == std::string("rbio")) {
        /* rbio is like bior, only with switched filters */
        std::string wv_nm;
        {
            std::stringstream ss;
            ss << "bior";
            ss << order_;
            wv_nm = ss.str();
        }
        std::stringstream ss;
        auto wave = wavelet(wv_nm);
        *this = wave;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;
        name_ = name;

        std::swap<size_t>(dec_len_, rec_len_);
        std::swap<size_t>(dec_len_, rec_len_);
        std::swap<double *>(rec_lo_double_, dec_lo_double_);
        std::swap<double *>(rec_hi_double_, dec_hi_double_);

        {
            size_t i, j;
            for (i = 0, j = rec_len_ - 1; i < j; i++, j--) {

                std::swap<double>(rec_lo_double_[i], rec_lo_double_[j]);
                std::swap<double>(rec_hi_double_[i], rec_hi_double_[j]);
                std::swap<double>(dec_lo_double_[i], dec_lo_double_[j]);
                std::swap<double>(dec_hi_double_[i], dec_hi_double_[j]);
            }
        }
        return;
    }

    /* Daubechies wavelets family */
    if (name == std::string("db")) {
        size_t coeffs_idx = order_ - 1;
        if (coeffs_idx >= ARRAYLEN(db_double))
            THROW_EX_FS(std::logic_error, "coeffs_idx >= ARRAYLEN(db_double)");
        auto wave = blank_discrete_wavelet(2 * order_);
        *this = wave;
        name_ = name;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;

        vanishing_moments_psi = order_;
        vanishing_moments_phi = 0;

        {
            size_t i;
            for (i = 0; i < rec_len_; ++i) {
                rec_lo_double_[i] = db_double[coeffs_idx][i];
                dec_lo_double_[i] = db_double[coeffs_idx][dec_len_ - 1 - i];
                rec_hi_double_[i] = ((i % 2) ? -1 : 1)
                                    * db_double[coeffs_idx][dec_len_ - 1 - i];
                dec_hi_double_[i] = (((dec_len_ - 1 - i) % 2) ? -1 : 1)
                                    * db_double[coeffs_idx][i];
            }
        }
    }

        /* Symlets wavelets family */
    else if (name == std::string("sym")) {
        size_t coeffs_idx = order_ - 2;
#ifdef WAVELET_TRANSFORM_DEBUGGING
        std::cout << "coeffs idx " << coeffs_idx << std::endl;
#endif
        if (coeffs_idx >= ARRAYLEN(sym_double))
            THROW_EX_FS(std::logic_error, "coeffs_idx >= ARRAYLEN(sym_double)");

        auto wave = blank_discrete_wavelet(2 * order_);
        *this = wave;
        name_ = name;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;

        {
            size_t i;
            for (i = 0; i < rec_len_; ++i) {
                rec_lo_double_[i] = sym_double[coeffs_idx][i];
                dec_lo_double_[i] = sym_double[coeffs_idx][dec_len_ - 1 - i];
                rec_hi_double_[i] = ((i % 2) ? -1 : 1)
                                    * sym_double[coeffs_idx][dec_len_ - 1 - i];
                dec_hi_double_[i] = (((dec_len_ - 1 - i) % 2) ? -1 : 1)
                                    * sym_double[coeffs_idx][i];
            }
        }
    }

        /* Coiflets wavelets family */
    else if (name == std::string("coif")) {
        size_t coeffs_idx = order_ - 1;
        if (coeffs_idx >= ARRAYLEN(coif_double))
            THROW_EX_FS(std::logic_error, "coeffs_idx >= ARRAYLEN(coif_double)");

        auto wave = blank_discrete_wavelet(6 * order_);
        *this = wave;
        name_ = name;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;

        {
            size_t i;
            for (i = 0; i < rec_len_; ++i) {
                rec_lo_double_[i] = coif_double[coeffs_idx][i] * sqrt2_double;
                dec_lo_double_[i] = coif_double[coeffs_idx][dec_len_ - 1 - i]
                                    * sqrt2_double;
                rec_hi_double_[i] = ((i % 2) ? -1 : 1)
                                    * coif_double[coeffs_idx][dec_len_ - 1 - i] * sqrt2_double;
                dec_hi_double_[i] = (((dec_len_ - 1 - i) % 2) ? -1 : 1)
                                    * coif_double[coeffs_idx][i] * sqrt2_double;
            }
        }
    }

        /* Biorthogonal wavelets family */
    else if (name == std::string("bior")) {
        unsigned N = order_ / 10, M = order_ % 10;
        size_t M_idx;
        size_t M_max;
        switch (N) {
            case 1:
                if (M % 2 != 1 || M > 5) THROW_EX_FS(std::logic_error, "M % 2 != 1 || M > 5");
                M_idx = M / 2;
                M_max = 5;
                break;
            case 2:
                if (M % 2 != 0 || M < 2 || M > 8) THROW_EX_FS(std::logic_error, "M % 2 != 0 || M < 2 || M > 8");
                M_idx = M / 2 - 1;
                M_max = 8;
                break;
            case 3:
                if (M % 2 != 1) THROW_EX_FS(std::logic_error, "M % 2 != 1");
                M_idx = M / 2;
                M_max = 9;
                break;
            case 4:
            case 5:
                if (M != N) THROW_EX_FS(std::logic_error, "M != N");
                M_idx = 0;
                M_max = M;
                break;
            case 6:
                if (M != 8) THROW_EX_FS(std::logic_error, "M != 8");
                M_idx = 0;
                M_max = 8;
                break;
            default:
                M_max = 0;
                THROW_EX_FS(std::logic_error, "bior wavelet order not supported.");
        }

        auto wave = blank_discrete_wavelet((N == 1) ? 2 * M : 2 * M + 2);
        *this = wave;
        name_ = name;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;

        {
            size_t n = M_max - M;
            for (size_t i = 0; i < rec_len_; ++i) {
                rec_lo_double_[i] = bior_double[N - 1][0][i + n];
                dec_lo_double_[i] = bior_double[N - 1][M_idx + 1][dec_len_ - 1 - i];
                rec_hi_double_[i] = ((i % 2) ? -1 : 1)
                                    * bior_double[N - 1][M_idx + 1][dec_len_ - 1 - i];
                dec_hi_double_[i] = (((dec_len_ - 1 - i) % 2) ? -1 : 1)
                                    * bior_double[N - 1][0][i + n];
            }
        }
    }

        /* Discrete FIR filter approximation of Meyer wavelet */
    else if (name == std::string("dmey")) {
        auto wave = blank_discrete_wavelet(62);
        *this = wave;
        name_ = name;
        wave.dec_lo_double_ = nullptr;
        wave.dec_hi_double_ = nullptr;
        wave.rec_lo_double_ = nullptr;
        wave.rec_hi_double_ = nullptr;

        {
            for (size_t i = 0; i < rec_len_; ++i) {
                rec_lo_double_[i] = dmey_double[i];
                dec_lo_double_[i] = dmey_double[dec_len_ - 1 - i];
                rec_hi_double_[i] = ((i % 2) ? -1 : 1)
                                    * dmey_double[dec_len_ - 1 - i];
                dec_hi_double_[i] = (((dec_len_ - 1 - i) % 2) ? -1 : 1)
                                    * dmey_double[i];
            }
        }
    } else {
        THROW_EX_FS(std::logic_error, "svr::wavelet_transform::wavelet::wavelet Unknown wavelet:" + name);
    }
}

spectral_transform::wavelet spectral_transform::wavelet::blank_discrete_wavelet(int filters_length)
{

    wavelet w;
    if (filters_length > 0) {
        w.dec_len_ = filters_length;
        w.rec_len_ = filters_length;
        w.dec_lo_double_ = reinterpret_cast<double *>(calloc(filters_length, sizeof(double)));
        w.dec_hi_double_ = reinterpret_cast<double *>(calloc(filters_length, sizeof(double)));
        w.rec_lo_double_ = reinterpret_cast<double *>(calloc(filters_length, sizeof(double)));
        w.rec_hi_double_ = reinterpret_cast<double *>(calloc(filters_length, sizeof(double)));
    } else {
        THROW_EX_FS(std::logic_error, "wavelet_transform::wavelet::blank_discrete_wavelet: filters_length == 0");
    }
    return w;
}

spectral_transform::wavelet::~wavelet()
{
    if (dec_lo_double_) free(dec_lo_double_);
    if (dec_hi_double_) free(dec_hi_double_);
    if (rec_lo_double_) free(rec_lo_double_);
    if (rec_hi_double_) free(rec_hi_double_);
}

void spectral_transform::wavelet::print() const
{
#ifdef WAVELET_TRANSFORM_DEBUGGING
                                                                                                                            std::cout << "wavelet_transform::wavelet::print():" << std::endl;
        std::cout << name_ << order_ << std::endl;
        std::cout << "dec_len_: " << dec_len_ << ", rec_len: " << rec_len_ << std::endl;
        
        std::cout << "dec_hi_double_: ";
        for (size_t i = 0; i < dec_len_; ++i)
        {
            std::cout << dec_hi_double_[i] << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "dec_lo_double_: ";
        for (size_t i = 0; i < dec_len_; ++i)
        {
            std::cout << dec_lo_double_[i] << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "rec_hi_double_: ";
        for (size_t i = 0; i < rec_len_; ++i)
        {
            std::cout << rec_hi_double_[i] << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "rec_lo_double_: ";
        for (size_t i = 0; i < rec_len_; ++i)
        {
            std::cout << rec_lo_double_[i] << ", ";
        }
        std::cout << std::endl;
#endif
}

}