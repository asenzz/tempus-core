#include "view/DatasetView.hpp"

namespace content
{


DatasetForm::DatasetForm ()
{

    name.id ("name");
    name.name ("name");
    name.message ("Dataset Name:");
    name.limits (1, 15);
    name.regex (booster::regex ("[\\w]+"));
    add (name);

    description.id ("description");
    description.name ("description");
    description.message ("Description:");
    description.limits (0, 150);
    add (description);

    lookback_time.id ("lookback_time");
    lookback_time.name ("lookback_time");
    lookback_time.message ("Lookback Time:");
    lookback_time.help ("Accepted format is " + time_duration_pattern);
    lookback_time.non_empty ();
    add (lookback_time);

    priority.add ("High");
    priority.add ("Above Normal");
    priority.add ("Normal");
    priority.add ("Below Normal");
    priority.add ("Low");
    priority.id ("priority");
    priority.name ("priority");
    priority.message ("Priority:");
    priority.selected (2);
    add (priority);

    svr_C.id ("svr_c");
    svr_C.name ("svr_c");
    svr_C.message ("svr_c Prarameter:");
    add (svr_C);

    svr_epsilon.id ("svr_epsilon");
    svr_epsilon.name ("svr_epsilon");
    svr_epsilon.message ("svr_epsilon Prarameter:");
    add (svr_epsilon);

    svr_kernel_param.id ("svr_kernel_param");
    svr_kernel_param.name ("svr_kernel_param");
    svr_kernel_param.message ("svr_kernel_param Prarameter:");
    add (svr_kernel_param);

    svr_kernel_param2.id ("svr_kernel_param2");
    svr_kernel_param2.name ("svr_kernel_param2");
    svr_kernel_param2.message ("svr_kernel_param2 Prarameter:");
    add (svr_kernel_param2);

    svr_decremental_distance.id ("svr_decremental_distance");
    svr_decremental_distance.name ("svr_decremental_distance");
    svr_decremental_distance.message ("svr_decremental_distance Prarameter:");
    add (svr_decremental_distance);

    svr_adjacent_levels_ratio.id ("svr_adjacent_levels_ratio");
    svr_adjacent_levels_ratio.name ("svr_adjacent_levels_ratio");
    svr_adjacent_levels_ratio.message ("svr_adjacent_levels_ratio Prarameter:");
    svr_adjacent_levels_ratio.high (1);
    svr_adjacent_levels_ratio.low (0);
    svr_adjacent_levels_ratio.error_message ("Must be between [0 - 1]");
    add (svr_adjacent_levels_ratio);

    transformation_levels.id ("transformation_levels");
    transformation_levels.name ("transformation_levels");
    transformation_levels.message ("Transformation Levels");
    transformation_levels.non_empty ();
    add (transformation_levels);

    for (const std::pair<std::string, std::string> &waveletNameAndId : get_wavelet_filters ()) {
        transformation_wavelet.add (waveletNameAndId.first, waveletNameAndId.second);
    }

    transformation_wavelet.id ("transformation_wavelet");
    transformation_wavelet.name ("transformation_wavelet");
    transformation_wavelet.message ("Transformation Wavelet Filter:");
    add (transformation_wavelet);

    submit.name ("save");
    submit.id ("save");
    submit.value ("Save");
    add (submit);

}

bool DatasetForm::validate ()
{

    return form::validate ()
           && validate_lookback_time ();

}

std::vector<std::pair<std::string, std::string>> DatasetForm::get_wavelet_filters ()
{
    std::vector<std::pair<std::string, std::string>> waveletFilters;

    waveletFilters.push_back (std::make_pair ("Haar", "haar"));

    // Daubechies : db1,db2,.., ,db15
    for (int i = 1; i <= 15; i++) {
        waveletFilters.push_back (std::make_pair ("(Daubechies) db" + std::to_string (i), "db" + std::to_string (i)));
    }

    // Biorthogonal: bior1.1 ,bior1.3 ,bior1.5 ,bior2.2 ,bior2.4 ,bior2.6 ,bior2.8 ,bior3.1 ,bior3.3 ,bior3.5 ,bior3.7 ,bior3.9 ,bior4.4 ,bior5.5 ,bior6.8
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior1.1", "bior1.1"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior1.3", "bior1.3"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior1.5", "bior1.5"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior2.2", "bior2.2"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior2.4", "bior2.4"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior2.6", "bior2.6"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior2.8", "bior2.8"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior3.1", "bior3.1"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior3.3", "bior3.3"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior3.5", "bior3.5"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior3.7", "bior3.7"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior3.9", "bior3.9"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior4.4", "bior4.4"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior5.5", "bior5.5"));
    waveletFilters.push_back (std::make_pair ("(Biorthogonal) bior6.8", "bior6.8"));

    // Coiflets: coif1,coif2,coif3,coif4,coif5
    for (int i = 1; i <= 5; i++) {
        waveletFilters.push_back (std::make_pair ("(Coiflets) coif" + std::to_string (i), "coif" + std::to_string (i)));
    }

    // Symmlets: sym1,sym2,........, sym20
    for (int i = 1; i <= 20; i++) {
        waveletFilters.push_back (std::make_pair ("(Symmlets) sym1" + std::to_string (i), "sym1" + std::to_string (i)));
    }

    return waveletFilters;
}

void DatasetWithForm::load_form_data ()
{
    object = ptr<svr::datamodel::Dataset> ();
    object->set_dataset_name (form.name.value ());
    object->set_description (form.description.value ());
    //    object->set_lookback_time(bpt::duration_from_string(form.lookback_time.value()));

    object->set_priority (form.priority.selected () == -1
                          ? svr::datamodel::Priority::Normal
                          : static_cast<svr::datamodel::Priority> (form.priority.selected ()));

    object->set_transformation_levels (form.transformation_levels.value ());
    object->set_gradients (form.gradients.value ());
    object->set_chunk_size (form.chunk_size.value ());
    object->set_multiout (form.chunk_size.value ());
}

bool DatasetForm::validate_lookback_time ()
{
    try {
        bpt::time_duration lookbackTime = bpt::duration_from_string (lookback_time.value ());

        if (lookbackTime < bpt::seconds (0)) {
            THROW_EX_FS(std::logic_error, "Lookback time must have positive value!");
        }
    } catch (const std::exception &e) {
        lookback_time.error_message (e.what ());
        lookback_time.valid (false);
        return false;
    }

    return true;
}
}
