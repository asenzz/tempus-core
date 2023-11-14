#pragma once

#include <vector>
#include <string>
#include <math.h>
#include <tgmath.h>
#include <cstring>
#include <algorithm>

extern "C"{
#include <wavelib/header/wavelib.h>
}

#include <spectral_transform.hpp>

/*
The class   modwt_base  implements MODWF transform with Fejecr - Korovkin filters.
 */
namespace svr   {

//Fejer-Korovkin filers order
   struct en_fk_filter_order {
       static int e_fk4;
       static int e_fk6;
       static int e_fk8;
       static int e_fk14;
       static int e_fk22;
   };


class modwt_transform : public spectral_transform
{
public:
    modwt_transform();
    modwt_transform(const size_t filter_order, const size_t decon_levels, const bool reversed_decon);
    modwt_transform(const std::string& filter, const size_t decon_levels);

    size_t get_filter_order() const;
    size_t get_num_decon_levels() const;
    void set_wavelen(const int wavelen, const bool enabled);

    void deconstruct1(int filter_order,
                                    int decon_level,
                                    const std::vector<double>& input,
                                    std::vector< std::vector<double> >& deconstructed);

    void reconstruct1(int filter_order,
                                   int decon_level,
                                   const std::vector< std::vector<double> >& deconstructed,
                                   std::vector<double>&  reconstructed);

    void deconstruct(int filter_order,
                                    int decon_level,
                                    const std::vector<double>& input,
                                    std::vector< std::vector<double> >& deconstructed);

    void reconstruct(int filter_order,
                                   int decon_level,
                                   const std::vector< std::vector<double> >& deconstructed,
                                   std::vector<double>&  reconstructed);

    virtual ~modwt_transform();

    virtual void transform(const std::vector<double>& input, std::vector<std::vector<double> >& deconstructed, size_t padding = 0) override;

    virtual void inverse_transform(const std::vector<double>& decon, std::vector<double>& recon, size_t padding = 0) const override;

    virtual size_t get_minimal_input_length(const size_t decremental_offset, const size_t lag_count) const;
private:
    const size_t _filer_order;
    const size_t _decon_levels;
};

}
