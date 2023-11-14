#include <iostream>
#include "deprecated/OnlineSMOSVR.hpp"

namespace svr {

// Learning Operations
size_t
OnlineSVR::forget(const vektor<ssize_t> &indexes)
{
    LOG4_BEGIN();

    // Initialization
    size_t flops = 0;
    if (!(indexes[0] >= 0 && indexes[indexes.size() - 1] < samples_trained_number))
        throw std::range_error("The indexes of the samples to remove are not valid.");
    for (decltype(indexes.size()) i = indexes.size() - 1; i >= 0; --i) flops += unlearn(indexes[i]);

    integrity_assured();

    LOG4_END();

    return flops;
}

size_t
OnlineSVR::forget(const vektor<double> &features)
{
    auto index = this->X->index_of(features);
    if (index < 0) throw std::range_error("Element to remove not exist!");
    return forget(index);
}


size_t
OnlineSVR::forget(const ssize_t sample_index)
{
    auto flops = unlearn(sample_index);
    integrity_assured();
    return flops;
}


size_t
OnlineSVR::unlearn(const ssize_t sample_index)
{
    LOG4_BEGIN();

    ++unlearn_calls;

    // Find the margin
    auto h = this->margin_inplace_all();
    ////////////////
    //
    // const double sample_margin = this->margin(sample_index);//(this->X[sample_index], this->Y[sample_index]);
    //    vektor<double> H;

    // If we have cached margins, load them to save compute effort.
    if (in_sample_margins.size() > 0) {
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Copying in_sample_margins -> H.");
 
        LOG4_DEBUG("Insample margin lenght: " << in_sample_margins.get_length());
        LOG4_DEBUG("Number samples trained: " << this->samples_trained_number);
#endif /* #ifdef ONLINE_SVR_DEBUG */
        if (in_sample_margins.size() != this->samples_trained_number)
            LOG4_ERROR("Incorrect number of in_sample_margins! Previous iteration number of margins "
                               << in_sample_margins.size()
                               << ", samples trained " << samples_trained_number);


        // TODO: change deep copy with a move or remove the redundant H entirely.
        in_sample_margins.copy_to(h);
        //H.add(sample_margin);
    } else {
        h.copy_to(in_sample_margins);
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("New calculated margin length is " << h.size());
#endif /* #ifdef ONLINE_SVR_DEBUG */
    }

    // const auto computed_sample_margin = this->margin(sample_index);
    // const auto insample_margin = h[sample_index];
    // Initializations
    size_t flops = 0;
    bool sample_removed = false;
    // Remaining set element
    auto sample_set_index = p_remaining_set_indexes->find(sample_index);
    if (sample_set_index >= 0) {
        /* Trivial solution */
        remove_from_remaining_set(sample_set_index);
        remove_sample(sample_index);
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Remove sample index " << sample_index << ", from remaining and exit");
        LOG4_FILE(ONLINESVR_LOG_FILE, "Remove sample index " << sample_index << ", from remaining and exit");
#endif /* #ifdef ONLINE_SVR_DEBUG */
        return ++flops;
    // Support set element
    } else if ((sample_set_index = p_support_set_indexes->find(sample_index)) >= 0) {
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Remove sample index " << sample_index << ", from support");
#endif /* #ifdef ONLINE_SVR_DEBUG */
       remove_from_support_set(sample_set_index);
#ifdef ONLINE_SVR_DEBUG
            LOG4_FILE(ONLINESVR_LOG_FILE,
                    "Remove sample index " << sample_index << ", from support");
#endif /* #ifdef ONLINE_SVR_DEBUG */

    // Error set element
    } else if ((sample_set_index = p_error_set_indexes->find(sample_index)) >= 0) {
        remove_from_error_set(sample_set_index);
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Remove sample index " << sample_index << ", from error");

            LOG4_FILE(ONLINESVR_LOG_FILE,
                    "Remove sample index " << sample_index << ", from error");
#endif /* #ifdef ONLINE_SVR_DEBUG */
    } else
        throw std::runtime_error(
                svr::common::formatter() << "Sample set for element " << sample_index << " not found!");

    // Main Loop
    while (!sample_removed) {
        // Check iterations number
        ++flops;
        if (flops > size_t(get_samples_trained_number() + 1) * online_iters_limit_mult ||
            flops > online_learn_iter_limit) {
            LOG4_ERROR("Iterations limit hit at " << flops << " iterations.");
            return flops;
        }
        if (flops % 100 == 0) LOG4_INFO("Working long ... ¯\\_(ツ)_/¯ iterations " << flops);

#ifdef ONLINE_SVR_DEBUG
        // KKT CONDITION CHECKING

        LOG4_DEBUG("Iteration " << flops << " finished, checking conditions.");

        // Check the KKTs weights
        if(this->violate_weights_support())
            LOG4_ERROR("Violated support set weights!");
        if(this->violate_weights_remaining())
            LOG4_ERROR("Violated remaining set weights!");
        if(this->violate_weights_error())
            LOG4_ERROR("Violated error set weights!");

        verify_kkt_conditions();
        // Check the KKTs margins
        if(this->violate_margin_support(h))
            LOG4_ERROR("Violated support set margins!");
        if(this->violate_margin_remaining(h))
            LOG4_ERROR("Violated remaining set margins!");
        if(this->violate_margin_error(h))
            LOG4_ERROR("Violated error set margins!");
#endif // ONLINE_SVR_DEBUG
        // Find Beta and Gamma
        auto beta = this->find_beta(sample_index);
        auto gamma = this->find_gamma(beta, sample_index);

        // Find Min Variation
        // TODO Make it adaptive to cost
        vektor<double> min_variations(number_variations);
        vektor<ssize_t> min_indexes(number_variations);
        vektor<ssize_t> flags(number_variations);
        find_unlearning_min_variations(h, beta, gamma, sample_index, number_variations, min_variations, min_indexes, flags);

#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("New function: number_variations: " << number_variations);
        LOG4_DEBUG("New function: min_variations: ");

        for(ssize_t i = 0; i < min_variations.size(); ++i)
            LOG4_DEBUG("min_index " << min_indexes[i] << ", min_variation " << min_variations[i] << ", flag " << flags[i]);
#endif // ONLINE_SVR_DEBUG

        // Find Min Variation
        const auto last_index = min_variations.size() - 1;
        auto min_variation = min_variations[last_index];
        auto flag = flags[last_index];
        auto min_index = min_indexes[last_index];
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("STARTING Iteration " << flops);
        //LOG4_FILE(ONLINESVR_LOG_FILE, "STARTING Iteration " << flops);
#endif // ONLINE_SVR_DEBUG

#ifdef ONLINE_SVR_DEBUG
        // LOG4_FILE(ONLINESVR_LOG_FILE, "iteration " << flops << " " << "before update weights and bias");
        
        LOG4_FILE(ONLINESVR_LOG_FILE, "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl <<
                                                "flag " << flag << std::endl <<
                                                "iteration " << flops << std::endl <<
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);

        LOG4_DEBUG(                              "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << flops << std::endl <<
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);
        
        
   
#endif // ONLINE_SVR_DEBUG
        // Update Weights and Bias
        update_weights_bias(h, beta, gamma, sample_index, min_variation);

#ifdef ONLINE_SVR_DEBUG
        LOG4_FILE(ONLINESVR_LOG_FILE, "iteration " << flops << " " << "after update weights and bias");
        LOG4_FILE(ONLINESVR_LOG_FILE, "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << flops << std::endl <<
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);

        LOG4_DEBUG(                              "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl <<
                                                "flag " << flag << std::endl <<
                                                "iteration " << flops << std::endl <<
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);
        
        
#endif // ONLINE_SVR_DEBUG

        // Move the Sample with Min Variaton to the New Set
        switch (flag) {
            // CASE 1: Remove the sample
            case 0:
                sample_removed = true;
                break;

            case 1:
                this->move_from_support_to_error_remaining_set(h, beta, gamma, min_index, min_variation);
                break;

                // CASE 2: Move Sample from Error Set to Support Set
            case 2:
                this->move_from_error_to_support_set(h, beta, gamma, min_index, min_variation);
                break;

                // CASE 3: Move Sample from RemainingSet to SupportSet
            case 3:
                this->move_from_remaining_to_support_set(h, beta, gamma, min_index, min_variation);
                break;

            default:
                throw std::runtime_error("Illegal flag value.");
        }
#ifdef ONLINE_SVR_DEBUG
        for(auto idx = 0; idx < h.get_length(); ++idx)
            LOG4_DEBUG("H [" << idx << "] = " << h[idx]);

        LOG4_DEBUG("FINISHING Iteration " << flops);
#endif // ONLINE_SVR_DEBUG
    }

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Iteration " << flops << " finished, checking conditions.");

    // Check the KKTs weights
    if(this->violate_weights_support())
        LOG4_ERROR("Violated support set weights!");
    if(this->violate_weights_remaining())
        LOG4_ERROR("Violated remaining set weights!");
    if(this->violate_weights_error())
        LOG4_ERROR("Violated error set weights!");

    verify_kkt_conditions();
    // Check the KKTs margins
    if(this->violate_margin_support(h))
        LOG4_ERROR("Violated support set margins!");
    if(this->violate_margin_remaining(h))
        LOG4_ERROR("Violated remaining set margins!");
    if(this->violate_margin_error(h))
        LOG4_ERROR("Violated error set margins!");

    LOG4_DEBUG("Updating margins. Previous size: " << in_sample_margins.get_length());
#endif // ONLINE_SVR_DEBUG
    h.copy_to(in_sample_margins);
    LOG4_DEBUG("Updated margins. New size is " << in_sample_margins.size());
    this->remove_sample(sample_index);

    return flops;
}

} // svr

