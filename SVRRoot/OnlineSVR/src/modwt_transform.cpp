#include <modwt_transform.hpp>
#include <util/math_utils.hpp>

#include <common/exceptions.hpp>

namespace svr {

int en_fk_filter_order::e_fk4 = en_fk4;
int en_fk_filter_order::e_fk6 = en_fk6;
int en_fk_filter_order::e_fk8 = en_fk8;
int en_fk_filter_order::e_fk14 = en_fk14;
int en_fk_filter_order::e_fk22 = en_fk22;

/*  valid FK filters for MODWT constructor call are : fk4, fk6, fk8, fk14, fk22
 */
modwt_transform::modwt_transform()
        : spectral_transform("fk8", 8),
          _filer_order(en_fk_filter_order::e_fk8),
          _decon_levels(8)
{}

modwt_transform::modwt_transform(const size_t filter_order, const size_t decon_levels,
                                 const bool reversed_decon = false)
        : spectral_transform("", decon_levels),
          _filer_order(filter_order),
          _decon_levels(decon_levels)
{}

modwt_transform::modwt_transform(const std::string &filter, const size_t decon_levels)
        : spectral_transform(filter, decon_levels),
          _filer_order(0),
          _decon_levels(decon_levels)
{}


modwt_transform::~modwt_transform()
{}

size_t modwt_transform::get_filter_order() const
{
    return _filer_order;
}

size_t modwt_transform::get_num_decon_levels() const
{
    return _decon_levels;
}

/*deconstruct1 and reconstruct1 were created with test purposes*/
void modwt_transform::deconstruct1(int filter_order,
                                   int decon_level,
                                   const std::vector<double> &input,
                                   std::vector<std::vector<double> > &deconstructed)
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

//        int wvlen = filter_order * pow(2, decon_level+1);
    int wvlen = input.size();

    double *inp_wave = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(filter_order);

    wt = wt_init1(wobj, wvlen, decon_level);

    //copy input data, todo: optimize later with modwt code
    std::memcpy(&inp_wave[0], &input[0], wvlen * sizeof(double));

    //deconstrust using modwt
    modwt1(wt, inp_wave);

//        std::vector<double> v;
//        for(int i=0; i < wt->outlength; ++i)
//            v.push_back(wt->output[i]);
//        deconstructed.push_back(v);

    //copy to output as required for train
    std::vector<double> v;
    for (int i = 0; i < wt->siglength; i++) {
        for (int k = 0; k <= wt->J; ++k) {
            v.push_back(wt->output[(k * (wt->siglength)) + i]);
        }
        deconstructed.push_back(v);
        v.clear();
    }

    free(inp_wave);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);

}

void modwt_transform::reconstruct1(int filter_order,
                                   int decon_level,
                                   const std::vector<std::vector<double> > &deconstructed,
                                   std::vector<double> &reconstructed)
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

//            int wvlen = filter_order * pow(2, decon_level+1);

    int wvlen = (int) deconstructed[0].size() / (decon_level + 1);

    double *recon_out = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(filter_order);

    wt = wt_init1(wobj, wvlen, decon_level);

    for (int i = 0; i < (int) deconstructed[0].size(); ++i)
        wt->output[i] = deconstructed[0][i];

    //do the inverse modwt, todo: optimize later with inverse modwt code
    inverse_modwt(wt, recon_out);

    //copy resut,   TODO reserve vector space before copy to it
    reconstructed.resize(wvlen);
    std::memcpy(&reconstructed[0], &recon_out[0], wvlen * sizeof(double));
    //  std::copy(&recon_out[0], &recon_outwvlen[wvlen-1], std::back_inserter(constructed));
    //  for(int i=0; i< wvlen; ++i)
    //        constructed.push_back(recon_out[i]);

    free(recon_out);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);

}

/*
 * modwt_transform::deconstruct and
 * modwt_transform::reconstruct
 * can be called with external parameters,
 * they are not related to the object of modwt_transform
 */
void modwt_transform::deconstruct(
        int filter_order,
        int decon_level,
        const std::vector<double> &input,
        std::vector<std::vector<double> > &deconstructed)
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

    int wvlen = filter_order * pow(2, decon_level + 1);

    double *inp_wave = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(filter_order);

    wt = wt_init1(wobj, wvlen, decon_level);

    //copy input data, todo: optimize later with modwt code
    std::memcpy(&inp_wave[0], &input[0], wvlen * sizeof(double));

    //deconstrust using modwt
    modwt1(wt, inp_wave);

    std::vector<double> v;
    for (int i = 0; i < wt->outlength; ++i)
        v.push_back(wt->output[i]);
    deconstructed.push_back(v);

    free(inp_wave);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);

}

void modwt_transform::reconstruct(int filter_order,
                                  int decon_level,
                                  const std::vector<std::vector<double> > &deconstructed,
                                  std::vector<double> &reconstructed)
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

    int wvlen = filter_order * pow(2, decon_level + 1);

    double *recon_out = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(filter_order);

    wt = wt_init1(wobj, wvlen, decon_level);

    for (int i = 0; i < (int) deconstructed[0].size(); ++i)
        wt->output[i] = deconstructed[0][i];

    //do the inverse modwt, todo: optimize later with inverse modwt code
    inverse_modwt(wt, recon_out);

    //copy resut,   TODO reserve vector space before copy to it
    reconstructed.resize(wvlen);
    std::memcpy(&reconstructed[0], &recon_out[0], wvlen * sizeof(double));
    //  std::copy(&recon_out[0], &recon_outwvlen[wvlen-1], std::back_inserter(constructed));
    //  for(int i=0; i< wvlen; ++i)
    //        constructed.push_back(recon_out[i]);

    free(recon_out);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);

}


/*  This implementation works as required from the calling code.
 */
void modwt_transform::transform(const std::vector<double> &input, std::vector<std::vector<double> > &deconstructed,
                                size_t padding)
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

    if (input.size() < svr::spectral_transform::modwt_levels_to_frame_length(_decon_levels, _filer_order)) {
        std::stringstream str;
        str << svr::common::color::modifier(svr::common::color::FG_RED)
            << "Not enough data for modwt_transform::transform."
            << "Provided: " << input.size()
            << " Expected: " << svr::spectral_transform::modwt_levels_to_frame_length(_decon_levels, _filer_order)
            << " values. " << svr::common::color::modifier(svr::common::color::FG_DEFAULT);
        throw (svr::common::insufficient_data(str.str()));
    }

//        int wvlen = _filer_order * pow(2, _decon_levels + 1);
    int wvlen = (int) input.size();

    double *inp_wave = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(_filer_order);

    wt = wt_init1(wobj, wvlen, _decon_levels);

    //copy input data, todo: optimize later with modwt code
    std::memcpy(&inp_wave[0], &input[0], wvlen * sizeof(double));

    //deconstrust using modwt
    modwt1(wt, inp_wave);

    //copy to output as required for train
    std::vector<double> v;
    for (int i = 0; i < wt->siglength; i++) {
        for (int k = 0; k <= wt->J; ++k) {
            v.push_back(wt->output[(k * (wt->siglength)) + i]);
        }
        deconstructed.push_back(v);
        v.clear();
    }

    free(inp_wave);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);
}

/*  This implementation works as required from the calling code.
 */
void
modwt_transform::inverse_transform(const std::vector<double> &decon, std::vector<double> &recon, size_t padding) const
{

    wave_object wobj = NULL;
    wt_object wt = NULL;

    if (decon.size() < _filer_order * pow(2, _decon_levels + 1)) {
        std::stringstream str;
        str << svr::common::color::modifier(svr::common::color::FG_RED)
            << "Not enough data for modwt_transform::inverse_transform."
            << "Provided: " << decon.size()
            << " Expected: " << svr::spectral_transform::modwt_levels_to_frame_length(_decon_levels, _filer_order)
            << " values. " << svr::common::color::modifier(svr::common::color::FG_DEFAULT);
        throw (svr::common::insufficient_data(str.str()));
    }

//        int wvlen = _filer_order * pow(2, _decon_levels + 1);
    int wvlen = (int) decon.size() / (_decon_levels + 1);

    double *recon_out = (double *) malloc(sizeof(double) * wvlen);

    wobj = wave_init1(_filer_order);

    wt = wt_init1(wobj, wvlen, _decon_levels);

    //copy decons signal for imodwt
    for (int i = 0; i < (int) decon.size(); ++i)
        wt->output[i] = decon[i];

    //do the inverse modwt, todo: optimize later with inverse modwt code
    inverse_modwt(wt, recon_out);

    recon.resize(wvlen);
    std::memcpy(&recon[0], &recon_out[0], wvlen * sizeof(double));

    free(recon_out);
    free(wt->params);
    wt_free(wt);
    wave_free(wobj);

}

size_t modwt_transform::get_minimal_input_length(const size_t decremental_offset, const size_t lag_count) const
{
    auto const decroffs = decremental_offset + lag_count;
    auto const fr_size = svr::spectral_transform::modwt_levels_to_frame_length(get_num_decon_levels(), get_filter_order());
    return std::max<size_t>(decroffs, fr_size) + svr::spectral_transform::modwt_residuals_length(get_num_decon_levels());
}


/*  These commented implementations were the original.
    They might be used by some time or their modifications.
 */
//void modwt_transform::transform(const std::vector<double>& input, std::vector<std::vector<double> >& deconstructed, int padding) const {
//
//        wave_object wobj = NULL;
//        wt_object wt = NULL;
//
//        int wvlen;   /* wvelen must be >= minimal wave length by modwt wave length fomula */
//
//        if( ! _use_custom_wavelen )
//            wvlen = (int)(_filer_order * powf(2, _decon_levels+2));
//        else
//            wvlen = _custom_wavelen;
//
//        double* inp_wave = (double*)malloc(sizeof(double)*wvlen);
//
//        wobj = wave_init1(_filer_order);
//
//        wt = wt_init1(wobj, wvlen, _decon_levels);
//
//        //copy input data, todo: optimize later with modwt code
//        std::memcpy(&inp_wave[0], &input[0], wvlen * sizeof(double));
//
//        //deconstrust using modwt
//        modwt1(wt, inp_wave);
//
//        //copy deconstructed signal into deconstructed
//        int lv_len = wt->outlength;
//
//        /*  copy deconstructed to output,
//         *  the inverse transform needs this order for reconstruct
//         *  the order in deconstructed is:
//         *  decomposition levels are in rows and base level is in the last row
//         *  for every row(level of deconstruction) we have the total number of deconstructed values
//         *  by the formula input_len * (deconstruction levels+1)
//         */
//        std::vector<double> v;
//        for(int i=0; i<wt->J+1; i++){
//            lv_len -= wt->siglength;
//            for(int k=0; k < wt->siglength; ++k) {
//                v.push_back(wt->output[lv_len + k]);
//            }
//            deconstructed.push_back(v);
//            v.clear();
//        }
//        /*  copy deconstructed to output,
//         *  the inverse transform needs this order for reconstruct
//         *  the order in deconstructed is:
//         *  we put on every row a vector with length = levels of decomposition + 1
//         */
////        std::vector<double> v;
////        for(int i=0; i < wt->siglength; i++){
////            for(int k=0; k < wt->J+1; ++k) {
////                v.push_back(wt->output[ (i*(wt->J+1)) + k]);
////            }
////            deconstructed.push_back(v);
////            v.clear();
////        }
//
//
//        //reverse deconstructed, base level is in the first row
//        if(_reversed_order)
//            std::reverse(deconstructed.begin(), deconstructed.end());
//
//
//        free(inp_wave);
//        free(wt->params);
//        wt_free(wt);
//        wave_free(wobj);
//}

//void modwt_transform::inverse_transform(const std::vector<double>& decon, std::vector<double>& recon, int padding) const {
//
//            wave_object wobj = NULL;
//            wt_object wt = NULL;
//
//            int wvlen;   /* wvelen must be >= minimal wave length by modwt wave length fomula */
//
//            if( ! _use_custom_wavelen )
//                wvlen = (int)(_filer_order * powf(2, _decon_levels+2));
//            else
//                wvlen = _custom_wavelen;
//
//            double* recon_out = (double*)malloc(sizeof(double)*wvlen);
//
//            wobj = wave_init1(_filer_order);
//
//            wt = wt_init1(wobj, wvlen, _decon_levels);
//
//            //copy decons signal for imodwt
//            std::memcpy(&wt->output[0], &decon[0], wvlen * sizeof(double));
//
//            if(_reversed_order)
//                std::reverse(&wt->output[0], &wt->output[wvlen]);
//
//            //do the inverse modwt, todo: optimize later with inverse modwt code
//
//            inverse_modwt(wt, recon_out);
//
//            recon.resize(wvlen);
//            std::memcpy(&recon[0], &recon_out[0], wvlen * sizeof(double));
//
//            free(recon_out);
//            free(wt->params);
//            wt_free(wt);
//            wave_free(wobj);
//
//}

} //end namespace svr
