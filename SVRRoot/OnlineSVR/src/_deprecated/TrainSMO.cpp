#include "deprecated/OnlineSMOSVR.hpp"
#include "deprecated/svm_batch.hpp"
#include "deprecated/smo.h"
#include "appcontext.hpp"

using namespace svr::common;

namespace svr {

void
OnlineSVR::stabilize_by_warm_batch_start()
{
    LOG4_BEGIN();

    // Prepare data for the batch trainer
    batch::svm_batch trainer(
            !svr_parameters.dont_update_r_matrix,
            svr_parameters.max_iter,
            svr_parameters.smo_epsilon_divisor);
    vektor<double> *p_weights_clone = this->p_weights->clone();
    auto rho = -this->bias;
    trainer.solve_epsilon_svr_warm_start(
            smo::smo_quadratic_matrix(*X, this->svr_parameters),
            *this->Y,
            *p_weights_clone,
            rho,
            *this);
    delete p_weights_clone;

    LOG4_END();
}

void
OnlineSVR::train_batch(const arma::mat &X, const arma::colvec &Y)
{
    svr::batch::svm_batch svm_batch;
    this->clear();
    svm_batch.train(X, Y, *this, false);
}

// Learning Operations
// TODO Optimize from here down the stack
size_t
OnlineSVR::train(const vmatrix<double> &X, const vektor<double> &Y)
{
    LOG4_BEGIN();

    // Initialization
    size_t flops = 0;
    // Learning
    for (auto i = 0; i < X.get_length_rows(); ++i) {
        // Element already trained
        const auto index = this->X->index_of(X.get_row_ptr(i));
        if (index > -1 && Y[i] == this->Y->get_value(index)) continue;
        // Training
        flops += this->learn(X.get_row_ref(i), Y[i]);
    }

    integrity_assured();

#if 0 /* ONLINE_SVR_DEBUG */
    // To be run when new etalon data is needed for train_predict_tests and/or the python onlinesvr program.
    if (std::abs(get_svr_parameters().get_svr_C() - 32.) < 1)
    {
        this->save_online_svr("../SVRRoot/OnlineSVR/test/test_data/online_svr_final_state.txt");
        this->X->save("../SVRRoot/OnlineSVR/test/test_data/x.txt");
        this->Y->save("../SVRRoot/OnlineSVR/test/test_data/y.txt");
    }
#endif

    LOG4_END();
    return flops;
}


size_t
OnlineSVR::train(const vektor<double> &features, const double label)
{
    return learn(features, label);
}


void
OnlineSVR::integrity_assured(const bool force)
{
    LOG4_BEGIN();
    if (!force && (!stabilized_learning || (learn_calls + unlearn_calls) % stabilize_iterations_count != 0)) return;

    if (verify_kkt_conditions()) {
        LOG4_DEBUG("Online model train is fully trained and stable.");
        return;
    }

    LOG4_WARN("KKT conditions after " << (learn_calls + unlearn_calls) << " modifications are not satisfied!");

    PROFILE_EXEC_TIME(stabilize_by_warm_batch_start(), "Stabilize online model train by warm start SMO scheme");
#ifdef ONLINE_SVR_DEBUG
    if (!this->verify_kkt_conditions())
        LOG4_ERROR("KKT conditions after online train -> warm smo stabilization are still not satisfied!");
    else
        LOG4_DEBUG("KKT conditions after online train -> warm smo start are stabilized and satisfied.");
#endif // #ifdef ONLINE_SVR_DEBUG

    LOG4_END();
}

// Learning Operations
size_t
OnlineSVR::train(
        const vmatrix<double> &features,
        const vektor<double> &labels,
        const vmatrix<double> &test_set_features,
        const vektor<double> &test_set_labels)
{
    // Initialization
    size_t flops = 0;
    LOG4_BEGIN();
    vektor<double> mean_errors;
    vektor<double> variances;
    vektor<double> predictions;
    // Learning
    for (auto i = 0; i < features.get_length_rows(); ++i) {
        // Training
        predictions.add(predict(features.get_row_ref(i)));
        flops += learn(features.get_row_ref(i), labels[i]);
        auto errors = margin(test_set_features, test_set_labels);
        mean_errors.add(errors.mean_abs());
        variances.add(errors.variance());
        integrity_assured();
    }
    // TODO Print out errors, variances and predictions

    LOG4_END();
    return flops;
}

// Learning Operations
size_t
OnlineSVR::train(
        const vmatrix<double> &X,
        const vektor<double> &Y,
        const size_t training_size,
        const size_t test_size)
{
    LOG4_BEGIN();

    // Initialization
    size_t flops = 0;
    vektor<double> test_errors;
    // Learning
    for (ssize_t i = 0; i < ssize_t(X.get_length_rows() - training_size - test_size + 1); ++i) {
        // Learning
        const auto training_set_x = X.extract_rows(i, i + training_size - 1);
        const auto *p_training_set_y = Y.extract(i, i + training_size - 1);
        const auto test_set_x = X.extract_rows(i + training_size, i + training_size + test_size - 1);
        const auto *p_test_set_y = Y.extract(i + training_size, i + training_size + test_size - 1);

        clear();
        train(training_set_x, *p_training_set_y);

        auto margins = margin(test_set_x, *p_test_set_y);
        test_errors.add(margins.mean_abs());

        delete p_training_set_y;
        delete p_test_set_y;
    }
    LOG4_END();

    return flops;
}


void
OnlineSVR::load_temporal_margin(const double sample_margin, vektor<double> &h)
{
    LOG4_BEGIN();

    // If we have cached margins, load them to save compute effort.
    if (in_sample_margins.size() > 0) {
        LOG4_DEBUG("Copying in_sample_margins -> H, and adding the margin " << sample_margin << " of the new point.");
        if (in_sample_margins.size() != this->samples_trained_number - 1)
            LOG4_ERROR(
                    "Incorrect number of in_sample_margins! Previous iteration number of margins "
                    << in_sample_margins.size() << ", samples trained previously " << samples_trained_number);
        // TODO change deep copy with a move or remove the redundant H entirely.
        in_sample_margins.copy_to(h);
        h.add(sample_margin);
    } else {
        LOG4_DEBUG("Calculating in_sample_margins.");
        h.resize(samples_trained_number);
        /*cilk_*/for (size_t i = 0; i < size_t(samples_trained_number); ++i) h[i] = margin(i);
        LOG4_DEBUG("New calculated margin length is " << h.size());
    }

    LOG4_END();
}


size_t
OnlineSVR::learn(const vektor<double> &features, const double label)
{
    LOG4_BEGIN();

    // Initializations
    X->add_row_copy(features);
    Y->add(label);
    p_weights->add(0.);

    ++samples_trained_number;
    ++learn_calls;

    if (this->save_kernel_matrix) this->add_to_kernel_matrix(features);
    else
        LOG4_WARN("Kernel matrix is not being saved!");

    size_t iterations = 0;
    bool new_sample_added = false;
    const auto svr_epsilon = svr_parameters.get_svr_epsilon();
    const auto sample_index = samples_trained_number - 1;
    const auto sample_margin = margin(features, label);

    vektor<double> h;
    load_temporal_margin(sample_margin, h);
#ifdef ONLINE_SVR_DEBUG
    LOG4_FILE(ONLINESVR_LOG_FILE, "Computed marign " << sample_margin);
#endif // #ifdef ONLINE_SVR_DEBUG

    // CASE 0: Right classified sample
    if (svr::common::less(std::abs((double)sample_margin), svr_epsilon)) {
        LOG4_DEBUG("The sample belongs to the remaining set, trivial solution.");
        // The margin is correct in this case, add it as it is.
        add_sample_to_remaining_set(sample_index);
        // If we have a cached margin vector, update it.
        // Otherwise, keep it empty - it will be fully created when needed.
        if (in_sample_margins.size() > 0) in_sample_margins.add(sample_margin);
        return ++iterations;
    }

    // TODO Task research can this loop be parallelized and to what extent?
    // Main loop
    while (!new_sample_added) {
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Iteration " << iterations << " finished, checking conditions.");

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

        
        // Check the KKTs weights
        this->violate_weights();
        // Check the KKTs margins
        this->violate_margin(h);

        // Check KKT original
        verify_kkt_conditions();

        LOG4_DEBUG("Online learn iteration " << ++iterations);
#endif // ONLINE_SVR_DEBUG

        // TODO Function which determines the upper limit of the possible point moves
        if (iterations > size_t(this->get_samples_trained_number() + 1) * online_iters_limit_mult ||
            iterations > online_learn_iter_limit) {
            LOG4_ERROR("Hit learning iterations limit " << iterations);
            break;
        }
        if (iterations % 100 == 0) LOG4_INFO("Working long ... ¯\\_(ツ)_/¯ iterations " << iterations);

        // Find p_beta and p_gamma
        vektor<double> beta = this->find_beta(sample_index);
        vektor<double> gamma = this->find_gamma(beta, sample_index);

        // Testing many variations functionality
        // Find Min Variation
        const ssize_t number_variations = max_variations;
        vektor<double> min_variations(number_variations);
        vektor<ssize_t> min_indexes(number_variations);
        vektor<ssize_t> flags(number_variations);

        find_learning_min_variations(
                h, beta, gamma, sample_index, number_variations, min_variations, min_indexes, flags);
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("New function: number_variations: " << max_variations);
        LOG4_DEBUG("New function: min_variations: ");
        for(ssize_t i = 0; i < min_variations.size(); ++i)
            LOG4_DEBUG("min_index " << min_indexes[i] << ", min_variation " << min_variations[i] << ", flag " << flags[i]);
#endif /* #ifdef ONLINE_SVR_DEBUG */

        LOG4_DEBUG("min_variations.size() " << min_variations.size() << "; MAX_VARIATIONS " << max_variations);
        // CASE 0: Right classified sample
        if (min_variations.size() == number_variations) {
            LOG4_DEBUG("The sample artificially to the remaining set, trivial solution.");
            // The margin is correct in this case, add it as it is.
            add_sample_to_remaining_set(sample_index);
            // If we have a cached margin vector, update it.
            // Otherwise, keep it empty - it will be fully created when needed.
            if (in_sample_margins.size() > 0) in_sample_margins.add(sample_margin);
            return ++iterations;
        }



        // Find Min Variation
        auto last_index = min_variations.size() - 1;
        auto min_variation = min_variations[last_index];
        auto flag = flags[last_index];
        auto min_index = min_indexes[last_index];
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Checking just after find learning variation.");
        verify_kkt_conditions();

        LOG4_DEBUG("H0 " << h[0] << ", beta0 " << beta[0] << ", gamma0 " << gamma[0] << ", w0 " << p_weights->get_value(0) <<
                         ", sample index " << sample_index << ", flag " << flag << ", iteration " << iterations << ", min variation " << min_variation << ", min index " << min_index);

        LOG4_FILE(ONLINESVR_LOG_FILE, "iteration " << iterations << " " << "before update weights and bias");
        
        LOG4_FILE(ONLINESVR_LOG_FILE, "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << iterations << std::endl << 
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);

        LOG4_DEBUG(                              "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << iterations << std::endl << 
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);
#endif // ONLINE_SVR_DEBUG
        update_weights_bias(h, beta, gamma, sample_index, min_variation);
#ifdef ONLINE_SVR_DEBUG
        LOG4_FILE(ONLINESVR_LOG_FILE, "iteration " << iterations << " " << "after update weights and bias");
        LOG4_FILE(ONLINESVR_LOG_FILE, "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << iterations << std::endl << 
                                                "min variation " << min_variation << std::endl << 
                                                "min index " << min_index << std::endl);

        LOG4_DEBUG(                              "H0 " << h[sample_index] << std::endl <<
//                                                "beta0 " << beta[0] << std::endl <<
//                                                "gamma0 " << gamma[0] << std::endl << 
                                                "w0 " << p_weights->get_value(sample_index) << std::endl <<
                                                "sample index " << sample_index << std::endl << 
                                                "flag " << flag << std::endl <<
                                                "iteration " << iterations << std::endl << 
                                                "min variation " << min_variation << std::endl <<
                                                "min index " << min_index << std::endl);
        LOG4_DEBUG("Checking just after update weights and bias.");
        verify_kkt_conditions();

        LOG4_DEBUG("H0 " << h[0] << ", beta0 " << beta[0] << ", gamma0 " << gamma[0] << ", w0 " << p_weights->get_value(0) <<
                         ", sample index " << sample_index << ", flag " << flag << ", iteration " << iterations << ", min variation " << min_variation << ", min index " << min_index);
#endif // ONLINE_SVR_DEBUG
        // Move the Sample with Min Variaton to the New Set
        switch (flag) {
            // CASE 1: Add the sample to the support set
            case 1:
                this->add_to_support_set(h, beta, gamma, sample_index, min_variation);
                new_sample_added = true;
                // LOG4_DEBUG("Added to support, flag " << flag);
                break;

                // CASE 2: Add the sample to the error set
            case 2:
                this->add_to_error_set(sample_index, min_variation);
                new_sample_added = true;
                // LOG4_DEBUG("Added to error, flag " << flag);
                break;

                // CASE 3: Move Sample from SupportSet to ErrorSet/RemainingSet
            case 3:
                this->move_from_support_to_error_remaining_set(h, beta, gamma, min_index, min_variation);
                break;

                // CASE 4: Move Sample from ErrorSet to SupportSet
            case 4:
                this->move_from_error_to_support_set(h, beta, gamma, min_index, min_variation);
                break;

                // CASE 5: Move Sample from RemainingSet to SupportSet
            case 5:
                this->move_from_remaining_to_support_set(h, beta, gamma, min_index, min_variation);
                break;

            default:
                throw std::range_error(formatter() << "Illegal flag value " << flag);
        }
    }

    // TODO: Remove H or change to move operation
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Last inner check of KKT");

    LOG4_DEBUG("Iteration " << iterations << " finished, checking conditions.");

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

    // Check the KKTs weights
    this->violate_weights();

    // Check the KKTs margins
    this->violate_margin(this->in_sample_margins);
#endif // ONLINE_SVR_DEBUG
    h.copy_to(this->in_sample_margins); // TODO Important leave commented out!

    LOG4_END();

    return ++iterations;
}

} // svr
