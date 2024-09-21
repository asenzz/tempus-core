/*
 * sw_transform.cpp
 *
 *  Port and some improvements of pywavelets library
 *  This replaces sgeorgiev's adapted wavelib by Rafat Hussain.
 *
 *  Automatic padding has been implemented to serve two purposes:
 *  1. Higher stationarity of deconstruction coefficients (since we will regress those for different time scales).
 *  2. The best found strategy for interpolation: padding_strategy::constant_value_at_end
 *     This strategy pads input_queue values with the last observed value, cuts the extra coefficients (not observed in input_queue given times),
 *     and pads decon_queue values of non-observed times with zeros on both ends.
 *  3. Inaccuracies of deconstruction and reconstruction happen near the boundaries, and the boundary effects increase with number of levels used.
 *  4. Using the padding strategies, the signal is deconvoluted only once instead of being split, so we get effects only in two regions of the signal (at the boundaries).
 *
 *
 */

// #define WAVELET_TRANSFORM_DEBUGGING


#ifdef WAVELET_TRANSFORM_DEBUGGING
#include <iterator>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <fstream>
#endif //WAVELET_TRANSFORM_DEBUGGING



#include "sw_transform.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>



namespace svr {

//CLASS SW_TRANSFORM

sw_transform::sw_transform(const std::string& name, const size_t levels, sw_transform::padding_strategy swt_padding_strategy)
: spectral_transform(name, levels), wavelet_(name), swt_padding_strategy_(swt_padding_strategy)
{
#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "sw_transform constructed" << wavelet_.name() <<wavelet_.order() << std::endl;
#endif
}

sw_transform::~sw_transform()
{
}

void sw_transform::double_downsampling_convolution_periodization_cls(CPTR(double)  input, const size_t N,
                                                       CPTR(double)  filter, const size_t F,
                                                       double * const output, const size_t step,
                                                       const size_t fstep) const
{
    size_t i = F/2, o = 0;
    const size_t padding = (step - (N % step)) % step;

    for (; i < F && i < N; i += step, ++o) {
        double sum = 0;
        size_t j;
        size_t k_start = 0;
        for (j = 0; j <= i; j += fstep)
            sum += filter[j] * input[i-j];
        if (fstep > 1)
            k_start = j - (i + 1);
        while (j < F){
            size_t k;
            for (k = k_start; k < padding && j < F; k += fstep, j += fstep)
                sum += filter[j] * input[N-1];
            for (k = k_start; k < N && j < F; k += fstep, j += fstep)
                sum += filter[j] * input[N-1-k];
        }
        output[o] = sum;
    }

    for(; i < N; i+=step, ++o){
        double sum = 0;
        size_t j;
        for(j = 0; j < F; j += fstep)
            sum += input[i-j]*filter[j];
        output[o] = sum;
    }

    for (; i < F && i < N + F/2; i += step, ++o) {
        double sum = 0;
        size_t j = 0;
        size_t k_start = 0;
        while (i-j >= N){
            size_t k;
            // for simplicity, not using fstep here
            for (k = 0; k < padding && i-j >= N; ++k, ++j)
                sum += filter[i-N-j] * input[N-1];
            for (k = 0; k < N && i-j >= N; ++k, ++j)
                sum += filter[i-N-j] * input[k];
        }
        if (fstep > 1)
            j += (fstep - j % fstep) % fstep;  // move to next non-zero entry
        for (; j <= i; j += fstep)
            sum += filter[j] * input[i-j];
        if (fstep > 1)
            k_start = j - (i + 1);
        while (j < F){
            size_t k;
            for (k = k_start; k < padding && j < F; k += fstep, j += fstep)
                sum += filter[j] * input[N-1];
            for (k = k_start; k < N && j < F; k += fstep, j += fstep)
                sum += filter[j] * input[N-1-k];
        }
        output[o] = sum;
    }

    for(; i < N + F/2; i += step, ++o){
        double sum = 0;
        size_t j = 0;
        while (i-j >= N){
            // for simplicity, not using fstep here
            size_t k;
            for (k = 0; k < padding && i-j >= N; ++k, ++j)
                sum += filter[i-N-j] * input[N-1];
            for (k = 0; k < N && i-j >= N; ++k, ++j)
                sum += filter[i-N-j] * input[k];
        }
        if (fstep > 1)
            j += (fstep - j % fstep) % fstep;  // move to next non-zero entry
        for (; j < F; j += fstep)
            sum += filter[j] * input[i-j];
        output[o] = sum;
    }
}

void sw_transform::double_upsampling_convolution_valid_sf_periodization_cls(CPTR(double) input, const size_t N,
                                                                     CPTR(double)  filter, const size_t F,
                                                                     double * const output, const size_t O) const
{
    // TODO? Allow for non-2 step

    size_t const start = F/4;
    size_t i = start;
    size_t const end = N + start - (((F/2)%2) ? 0 : 1);
    size_t o = 0;

    if(F%2) std::logic_error("sw_transform::double_upsampling_convolution_valid_sf_periodization: Filter must have even-length.");

    if ((F/2)%2 == 0){
        // Shift output one element right. This is necessary for perfect reconstruction.

        // i = N-1; even element goes to output[O-1], odd element goes to output[0]
        size_t j = 0;
        while(j <= start-1){
            size_t k;
            for (k = 0; k < N && j <= start-1; ++k, ++j){
                output[2*N-1] += filter[2*(start-1-j)] * input[k];
                output[0] += filter[2*(start-1-j)+1] * input[k];
            }
        }
        for (; j <= N+start-1 && j < F/2; ++j){
            output[2*N-1] += filter[2*j] * input[N+start-1-j];
            output[0] += filter[2*j+1] * input[N+start-1-j];
        }
        while (j < F / 2){
            size_t k;
            for (k = 0; k < N && j < F/2; ++k, ++j){
                output[2*N-1] += filter[2*j] * input[N-1-k];
                output[0] += filter[2*j+1] * input[N-1-k];
            }
        }

        o += 1;
    }

    for (; i < F/2 && i < N; ++i, o += 2){
        size_t j = 0;
        for(; j <= i; ++j){
            output[o] += filter[2*j] * input[i-j];
            output[o+1] += filter[2*j+1] * input[i-j];
        }
        while (j < F/2){
            size_t k;
            for(k = 0; k < N && j < F/2; ++k, ++j){
                output[o] += filter[2*j] * input[N-1-k];
                output[o+1] += filter[2*j+1] * input[N-1-k];
            }
        }
    }

    for (; i < N; ++i, o += 2){
        size_t j;
        for(j = 0; j < F/2; ++j){
            output[o] += filter[2*j] * input[i-j];
            output[o+1] += filter[2*j+1] * input[i-j];
        }
    }

    for (; i < F/2 && i < end; ++i, o += 2){
        size_t j = 0;
        while(i-j >= N){
            size_t k;
            for (k = 0; k < N && i-j >= N; ++k, ++j){
                output[o] += filter[2*(i-N-j)] * input[k];
                output[o+1] += filter[2*(i-N-j)+1] * input[k];
            }
        }
        for (; j <= i && j < F/2; ++j){
            output[o] += filter[2*j] * input[i-j];
            output[o+1] += filter[2*j+1] * input[i-j];
        }
        while (j < F / 2){
            size_t k;
            for (k = 0; k < N && j < F/2; ++k, ++j){
                output[o] += filter[2*j] * input[N-1-k];
                output[o+1] += filter[2*j+1] * input[N-1-k];
            }
        }
    }

    for (; i < end; ++i, o += 2){
        size_t j = 0;
        while(i-j >= N){
            size_t k;
            for (k = 0; k < N && i-j >= N; ++k, ++j){
                output[o] += filter[2*(i-N-j)] * input[k];
                output[o+1] += filter[2*(i-N-j)+1] * input[k];
            }
        }
        for (; j <= i && j < F/2; ++j){
            output[o] += filter[2*j] * input[i-j];
            output[o+1] += filter[2*j+1] * input[i-j];
        }
    }
}

void sw_transform::double_upsampling_convolution_valid_sf_cls(CPTR(double)  input, const size_t N,
                                                CPTR(double)  filter, const size_t F,
                                                double * const output, const size_t O) const
{
    double_upsampling_convolution_valid_sf_periodization_cls(input, N, filter, F, output, O);
}

/* basic SWT step (TODO: optimize) */
void sw_transform::double_swt_cls(
        CPTR(double)  input,
        const size_t input_len,
        CPTR(double)  filter,
        const size_t filter_len,
        double *const output,
        const size_t output_len,
        const size_t level) const
{
    double *e_filter;
    size_t i, e_filter_len, fstep;

    if(level < 1)
        throw std::logic_error("level < 1");

    if(level > swt_max_level(input_len))
        throw std::logic_error("level > swt_max_level(input_len)");

    if(output_len != input_len)
        throw std::logic_error("output_len != swt_buffer_length(input_len)");

    /* TODO: quick hack, optimize */
    if (level > 1) {
        /* allocate filter first */
        e_filter_len = filter_len << (level - 1);
        e_filter = reinterpret_cast<double*>(calloc(e_filter_len, sizeof(double)));
        if(e_filter == NULL) throw std::logic_error("e_filter == NULL");
        fstep = 1 << (level - 1);  // spacing between non-zero filter entries

        /* compute upsampled filter values */
        for(i = 0; i < filter_len; ++i) e_filter[i << (level-1)] = filter[i];

        double_downsampling_convolution_periodization_cls(
                input, input_len, e_filter,
                e_filter_len, output, 1,
                fstep);
        free(e_filter);
        return;

    } else {
        double_downsampling_convolution_periodization_cls(input, input_len, filter,
                                                    filter_len, output, 1,
                                                    1);
        return;
    }
}

/* Details at specified level
 * input - approximation coeffs from upper level or signal if level == 1
 */
void sw_transform::double_swt_d_cls(CPTR(double)  input, const size_t input_len,
                                    const spectral_transform::wavelet &wavelet,
                                    double *const output, const size_t output_len,
                                    const size_t level) const{
    double_swt_cls(input, input_len, wavelet.dec_hi_double(),
                            wavelet.dec_len(), output, output_len, level);
}

void sw_transform::double_swt_a_cls(CPTR(double)  input, const size_t input_len,
                                    const sw_transform::wavelet &wavelet,
                                    double *const output, const size_t output_len, const size_t level) const
{
    double_swt_cls(input, input_len, wavelet.dec_lo_double(),
                            wavelet.dec_len(), output, output_len, level);
}

void sw_transform::double_idwt_cls(CPTR(double)  coeffs_a, const size_t coeffs_a_len,
                     CPTR(double)  coeffs_d, const size_t coeffs_d_len,
                     double * const output, const size_t output_len,
                     const spectral_transform::wavelet& wavelet) const
{
    size_t input_len;
    if(coeffs_a != NULL && coeffs_d != NULL){
        if(coeffs_a_len != coeffs_d_len)
            throw std::logic_error("sw_transform::double_idwt: coeffs_a_len != coeffs_d_len");
        input_len = coeffs_a_len;
    } else if(coeffs_a != NULL){
        input_len  = coeffs_a_len;
    } else if (coeffs_d != NULL){
        input_len = coeffs_d_len;
    } else {
        throw std::logic_error("sw_transform::double_idwt");
    }

    /* check output size */
    if(output_len != input_len * 2)
        throw std::logic_error("sw_transform::double_idwt: output_len != idwt_buffer_length(input_len, wavelet->rec_len, mode)");

    /*
     * Set output to zero (this can be omitted if output array is already
     * cleared) memset(output, 0, output_len * sizeof(TYPE));
     */

    /* reconstruct approximation coeffs with lowpass reconstruction filter */
    if(coeffs_a){
        double_upsampling_convolution_valid_sf_cls(coeffs_a, input_len,
                                                  wavelet.rec_lo_double(),
                                                  wavelet.rec_len(), output,
                                                  output_len);
    }
    /*
     * Add reconstruction of details coeffs performed with highpass
     * reconstruction filter.
     */
    if(coeffs_d){
        double_upsampling_convolution_valid_sf_cls(coeffs_d, input_len,
                                                  wavelet.rec_hi_double(),
                                                  wavelet.rec_len(), output,
                                                  output_len);
    }
}

unsigned char
sw_transform::swt_max_level(const size_t input_len)
{
    /* check how many times input_len is divisible by 2 */
    auto l = input_len;
    unsigned char j = 0;
    while (l > 0){
        if (l % 2) return j;
        l /= 2;
        j++;
    }
    return j;
}

void sw_transform::fill_swt_data_structure(std::vector<double> input, std::vector<double> &padded_data_container,
                                           const size_t padding, size_t &out_coercion_padding) const
{
    auto n_data = input.size();
    if (swt_padding_strategy_ == sw_transform::padding_strategy::antisymmetric && static_cast<size_t>(padding) != input.size())
    {
        if (static_cast<size_t>(padding) == 0)
        {
            if (n_data % static_cast<int>(pow(2,levels_)) != 0)
            {
                throw std::logic_error("sw_transform::fill_swt_data_structure: padding hard set to 0, but coercion padding would be nonzero.");
            }
            else
            {
                out_coercion_padding = 0;
                padded_data_container.resize(input.size());
                std::copy_n(input.begin(), input.size(), padded_data_container.begin());
                return;
            }
        }
        throw std::logic_error("sw_transform::fill_swt_data_structure: swt_padding_strategy_ == sw_transform::padding_strategy::antisymmetric && padding != input.size(): this case is not implemented.");
    }
    if (padding > 0)
    {
        // Insert front padding
        switch (swt_padding_strategy_) {
            case sw_transform::padding_strategy::constant_value_at_end:
            case sw_transform::padding_strategy::first_order_simple_approximation:
            case sw_transform::padding_strategy::antisymmetric:
            case sw_transform::padding_strategy::symmetric:
                padded_data_container.insert(padded_data_container.end(), padding, input[0]);
                break;
            default:
                throw std::logic_error("sw_transform::fill_swt_data_structure: invalid padding strategy");
        }

        // Insert the actual data
        padded_data_container.insert(padded_data_container.end(), input.begin(), input.end());

        // Insert back padding
        switch (swt_padding_strategy_) {
            case sw_transform::padding_strategy::constant_value_at_end:
            case sw_transform::padding_strategy::first_order_simple_approximation:
            case sw_transform::padding_strategy::antisymmetric:
            case sw_transform::padding_strategy::symmetric:
                padded_data_container.insert(padded_data_container.end(), padding, input[input.size()-1]);
                break;
            default:
                throw std::logic_error("sw_transform::fill_swt_data_structure: invalid padding strategy");
        }
        n_data = padded_data_container.size();
    }
    else
    {
        padded_data_container.insert(padded_data_container.end(), input.begin(), input.end());
        n_data = padded_data_container.size();
    }
#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "will do coercion padding now, nData="<<nData<<", pow(2,levels_)= "<< static_cast<int>(pow(2,levels_))<< std::endl;
#endif
    // coerce the input signal to be an integer multiple of 2^(nLevels), by padding to the right.
    if (n_data % static_cast<int>(pow(2,levels_)) != 0)
    {
        #ifdef WAVELET_TRANSFORM_DEBUGGING
        std::cout << "will do coercion padding now" << std::endl;
        #endif
        out_coercion_padding = pow(2,levels_) - n_data % static_cast<int>(pow(2,levels_));
        switch (swt_padding_strategy_) {
            case sw_transform::padding_strategy::constant_value_at_end:
            case sw_transform::padding_strategy::first_order_simple_approximation:
            case sw_transform::padding_strategy::antisymmetric:
            case sw_transform::padding_strategy::symmetric:
                padded_data_container.insert(padded_data_container.end(), out_coercion_padding, input[input.size()-1]);
                break;
            default:
                throw std::logic_error("sw_transform::fill_swt_data_structure: invalid padding strategy");
        }
#ifdef WAVELET_TRANSFORM_DEBUGGING
        std::cout << "Padded " << out_coercion_padding << " values to the right, to coerce the signal to be N * 2^(nLevels), ie: " << padded_data_container.size() << std::endl;
#endif
    }
    else
    {
        out_coercion_padding = 0;
    }
    if (swt_padding_strategy_ == sw_transform::padding_strategy::first_order_simple_approximation)
    {
        // write the slope and coefficients
        double slope = (input[input.size() - 1] - input[0]) / input.size();
        double intercept = input[0] - slope * padding;
        for (size_t i = 0; i < padding; ++i)
        {
            padded_data_container[i] = slope * i + intercept;
        }
        for (size_t i = padding+input.size(); i < padded_data_container.size(); ++i)
        {
            padded_data_container[i] = slope*i + intercept;
        }
    }
    else if (swt_padding_strategy_ == sw_transform::padding_strategy::antisymmetric)
    {      
        for (size_t i = 0; i < padding; ++i)
        {
            // Left boundary
            padded_data_container[padding - i - 1] = 2 * input[0] - input[i];
            // Right boundary
            padded_data_container[padding + input.size() + i] = 2 * input[input.size()-1] - input[input.size()-1-i];
            // The remaining coerced_padding coefficients shall be zero for now.
            // It is not a problem unless we use the absolute max of levels possible for the signal.
        }
#ifdef WAVELET_TRANSFORM_DEBUGGING
        std::ofstream f("antisym.txt");
        for (size_t i = 0; i < padded_data_container.size(); ++i)
            f << padded_data_container[i] << "\n";
        f.close();
#endif
    }
    else if (swt_padding_strategy_ == sw_transform::padding_strategy::symmetric)
    {
        for (size_t i = 0; i < padding; ++i)
        {
            padded_data_container[padding - i - 1] = input[i];
            padded_data_container[padding + input.size() + i] = input[input.size()-1-i];
        }
#ifdef WAVELET_TRANSFORM_DEBUGGING
        std::ofstream f("sym.txt");
        for (size_t i = 0; i < padded_data_container.size(); ++i)
            f << padded_data_container[i] << "\n";
        f.close();
#endif
    }
}

void sw_transform::transform(const std::vector<double>& input, std::vector<std::vector<double> >& result, const size_t padding)
{
    if (padding < 0)
    {
        throw std::logic_error("sw_transform::swt: was given padding < 0!");
    }
    // TODO: push padding down on the call stack
    if (static_cast<int>(pow(2,levels_)) > (int)input.size())
    {
        throw std::logic_error( std::string("sw_transform::swt: Level value ") + std::to_string(levels_) + std::string("too high - max level for current data size is ") + std::string(std::to_string(swt_max_level(input.size()))));
    }

    std::vector<double> padded_data_container;
    size_t coercion_padding;
    fill_swt_data_structure(input, padded_data_container, padding, coercion_padding);
    auto n_data = padded_data_container.size();
    double* inputarr = padded_data_container.data();

#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "padded data size: " << nData <<", max number of levels: "<<(unsigned int)swt_max_level(nData)<< std::endl;
#endif

    auto output_len = n_data;
    
#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "output_length: " << output_len << std::endl;
#endif

    std::vector<std::pair<std::vector<double>,std::vector<double> > > res;
    for (size_t cur_level = 1; cur_level <= levels_; ++cur_level)
    {       
        res.push_back(std::pair<std::vector<double>,std::vector<double> >());
        res[res.size()-1].first.resize(output_len);
        res[res.size()-1].second.resize(output_len);
        double* cA = res[res.size()-1].first.data();
        double* cD = res[res.size()-1].second.data();
        double_swt_a_cls(inputarr, n_data, wavelet_, cA, output_len, cur_level);
        double_swt_d_cls(inputarr, n_data, wavelet_, cD, output_len, cur_level);
        inputarr = cA;
    }

    // TODO: this reverse operation is costly, optimize.
    std::reverse(res.begin(), res.end());

#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "Padding " << padding << ", coercion_padding " << coercion_padding << std::endl;
#endif

    // Extract cA[0] and cD[all] as columns, while trimming the padded coefficients.
    result.clear();
    {
        result.resize(n_data - 2 * padding - coercion_padding, std::vector<double>(levels_ + 1));
#ifdef WAVELET_TRANSFORM_DEBUGGING    
        std::cout << "result shape: " << result.size() << ", " << result[0].size() << std::endl;        
        std::cout << "nData: " << nData << ", padding: "<<padding << ", coerced_padding: "<<coercion_padding<< std::endl;
#endif
        // Transform to time-major, level-minor format
        for (size_t row = padding; row < n_data - padding - coercion_padding; ++row)
        {
            result[row - padding][0] = res[0].first[row];
            for (size_t level = 0; level < levels_; ++level)
            {
                result[row - padding][level+1] = res[level].second[row];
            }
        }
    }

#ifdef WAVELET_TRANSFORM_DEBUGGING
    {
        std::cout << "DECON FULL" << std::endl;
        std::stringstream ss;
        ss << "swt-decon-queue-"<<unnecessary_output_suffix<<"-untrimmed.txt";
        std::ofstream output;
        output.open(ss.str());
        std::vector<std::vector<double>> tmp;
        tmp.resize(nData, std::vector<double>(levels_ + 1));
        for (size_t row = 0; row < nData; ++row)
        {
            tmp[row][0] = res[0].first[row];
            for (size_t level = 0; level < levels_; ++level)
            {
                tmp[row][level+1] = res[level].second[row];
            }
        }
        for (auto item : tmp) {
            for (auto item2 : item)
                output << item2 << " ";
            output << std::endl;
        }
        output.close();
    }
#endif
}

std::vector<double> sw_transform::idwt_single(std::vector<double> cA, std::vector<double> cD, const spectral_transform::wavelet& wavelet) const
{

    auto rec_len = 2 * cA.size();
    std::vector<double> result(rec_len);

#ifdef WAVELET_TRANSFORM_DEBUGGING
    assert(cA.size() == cD.size());
#endif
    double_idwt_cls(cA.data(), cA.size(), cD.data(), cD.size(), result.data(), result.size(), wavelet);
    return result;
}

void sw_transform::fill_iswt_data_structure(
        const std::vector<double>& frame,
        std::vector<std::pair<std::vector<double>, std::vector<double>>>& deconstructed,
        const size_t padding,
        const size_t coercion_padding) const
{
    auto orig_len_coef = frame.size() / (levels_ + 1);
    auto len_coef = padding + orig_len_coef + padding + coercion_padding;
    // Construct the data structure to use for calculations
    for (size_t level = 0; level < levels_; ++level)
    {
        deconstructed.push_back(std::pair<std::vector<double>,std::vector<double>>());
        if (level == 0)
        {
            deconstructed[deconstructed.size()-1].first.resize(len_coef);
        }
        deconstructed[deconstructed.size()-1].second.resize(len_coef);
    }
    // Copy values from the original frame to the middle of the new data structure
    for (size_t row = 0; row < orig_len_coef; ++row)
    {
        for (size_t level = 0; level < levels_; ++level)
        {
            if (level == 0)
            {
                // cA[0]
                deconstructed[level].first[row + padding] = frame[row];
            }
            // cD[level]
            deconstructed[level].second[row + padding] = frame[orig_len_coef*(level + 1) + row];
        }
    }
    std::vector<std::pair<double,double>> slopes_and_intercepts;
    
    for (size_t row = 0; row < padding; ++row)
    {
        for (size_t level = 0; level < levels_; ++level)
        {
            switch (swt_padding_strategy_)
            {
                // TODO: we do not use padding strategy right now with the iswt transform
                case sw_transform::padding_strategy::constant_value_at_end:
                case sw_transform::padding_strategy::first_order_simple_approximation:
                case sw_transform::padding_strategy::antisymmetric:
                case sw_transform::padding_strategy::symmetric:
                    if (level == 0)
                    {
                        // cA[0] will be padded left with constant zeros (the endpoint cA[0]s).
                        deconstructed[level].first[row] = frame[0];
                        deconstructed[level].first[padding + row + orig_len_coef] = frame[orig_len_coef - 1];
                    }
                    // cD[level] will be padded left and right with constant zeros
                    deconstructed[level].second[row] = 0;
                    deconstructed[level].second[padding + row + orig_len_coef] = 0;
                    break;
                default:
                    throw std::logic_error("void sw_transform::fill_iswt_data_structure: invalid padding strategy.");
            }
        }
    }
    // now pad with coerced_padding (to make sure we have N * 2^levels-length signal)
    for (size_t row = padding * 2 + orig_len_coef; row < len_coef; ++row)
    {
        for (size_t level = 0; level < levels_; ++level)
        {
            switch (swt_padding_strategy_)
            {
                case sw_transform::padding_strategy::constant_value_at_end:
                case sw_transform::padding_strategy::first_order_simple_approximation:
                case sw_transform::padding_strategy::antisymmetric:
                case sw_transform::padding_strategy::symmetric:
                    if (level == 0)
                    {
                        // cA will be padded by a constant value
                        deconstructed[level].first[row] = frame[orig_len_coef - 1];
                    }
                    // cD[level] will be padded with constant zeros regardless of strategy
                    deconstructed[level].second[row] = 0;
                    break;
                default:
                    throw std::logic_error("void sw_transform::fill_iswt_data_structure: invalid padding strategy.");
            }
        }
    }
}

void sw_transform::inverse_transform(const std::vector<double>& frame, std::vector<double>& result, const size_t padding) const
{
    // TODO: push padding down on the call stack

    // It may be faster or not faster to transform the frame back into a level-major format.
    // It is transformed here for code clarity.
    
    auto n_data = frame.size();
    size_t len_coef =  n_data / (levels_ + 1);
    
    if (static_cast<size_t>(pow(2,levels_)) > len_coef)
    {
        throw std::logic_error( std::string("sw_transform::swt: Level value ") + std::to_string(levels_) + std::string("too high - max level for current data size is ") + std::string(std::to_string(swt_max_level(len_coef))));
    }

    auto n_levels = levels_;
    size_t coercion_padding = 0;
    
    len_coef += 2 * padding;
    // coerce the input signal to be an integer multiple of 2^(nLevels), by padding to the right.
    if (len_coef % static_cast<int>(pow(2,levels_)) != 0)
    {
        coercion_padding = pow(2,levels_) - len_coef % static_cast<int>(pow(2,levels_));
        len_coef += coercion_padding;
#ifdef WAVELET_TRANSFORM_DEBUGGING
        std::cout << "coercion padding: " << coercion_padding;
#endif
    }

#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "len_coef: " << len_coef << std::endl;
#endif
    std::vector<std::pair<std::vector<double>, std::vector<double> > > deconstructed;
    fill_iswt_data_structure(frame, deconstructed, padding, coercion_padding);

#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "deconstructed shape: " << deconstructed[0].first.size() << std::endl;
    for (size_t level = 0; level < levels_; ++level)
    {
        const auto& cA = deconstructed[level].first;
        const auto& cD = deconstructed[level].second;
        {
            std::ofstream out(std::string("level")+std::to_string(level)+std::string("-cAcD-extrapolated")+".txt");
            std::ostream_iterator<double> out_iterator(out, "\n");
            std::copy(cA.begin(), cA.end(), out_iterator);
            std::copy(cD.begin(), cD.end(), out_iterator);
            out.close();
        }
    }
    std::cout << levels_ << " levels, " << len_coef << " coef length (cA[0] and each cD[n] in nLevels)" << std::endl;
#endif

    std::vector<double> cA(len_coef);
    std::copy_n(deconstructed[0].first.begin(), deconstructed[0].first.size(), cA.begin());
    
    for (auto j = n_levels; j > 0; --j)
    {
        auto step_size = (size_t) pow(2, j-1);
        auto last_idx = step_size;
        const auto& cD = deconstructed[n_levels - j].second;
        
        for (size_t first = 0; first < last_idx; ++first)
        {
            std::vector<size_t> idces;
            for (size_t i = first; i < cD.size(); i += step_size)
            {
                idces.push_back(i);
            }
            std::vector<size_t> even_idces;
            std::vector<size_t> odd_idces;
            for (size_t i = 0; i < idces.size() / 2; ++i)
            {
                even_idces.push_back(idces[2*i]);
                odd_idces.push_back(idces[2*i+1]);
            }

            std::vector<double> even_cA, odd_cA, even_cD, odd_cD;
#ifdef WAVELET_TRANSFORM_DEBUGGING
            assert(even_cA.size() == odd_cA.size());
#endif
            for (size_t i = 0; i < even_idces.size(); ++i)
            {
                even_cA.push_back(cA[even_idces[i]]);
                odd_cA.push_back(cA[odd_idces[i]]);
                even_cD.push_back(cD[even_idces[i]]);
                odd_cD.push_back(cD[odd_idces[i]]);
            }

            auto x1 = idwt_single(even_cA, even_cD, wavelet_);
            auto x2 = idwt_single(odd_cA, odd_cD, wavelet_);

            double buf = x2[x2.size()-1];
            for (size_t i = x2.size()-1; i > 0; --i)
            {
                x2[i] = x2[i-1];
            }
            x2[0] = buf;

            for (size_t idx = 0; idx < idces.size(); ++idx)
            {
                    cA[idces[idx]] = (x1[idx] + x2[idx])/2;
            }
        }
    }

#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "cA size: " << cA.size() << std::endl;
#endif

    result.resize(cA.size() - 2 * padding - coercion_padding);

    std::copy_n(cA.begin() + padding, cA.size() - 2 * padding - coercion_padding, result.begin());
#ifdef WAVELET_TRANSFORM_DEBUGGING
    std::cout << "result size: " << result.size() << std::endl;
#endif
}

size_t sw_transform::get_minimal_input_length(const size_t decremental_offset, const size_t lag_count) const
{
    return decremental_offset + lag_count;
}

}

#undef WAVELET_TRANSFORM_DEBUGGING
