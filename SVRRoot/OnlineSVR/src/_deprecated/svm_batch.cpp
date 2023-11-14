#include "common.hpp"
#include "deprecated/svm_batch.hpp"
#include "deprecated/smo.h"
#include "util/PerformanceUtils.hpp"
#include "appcontext.hpp"
#include "deprecated/csmosolver.hpp"

namespace svr {
namespace batch {

void cholesky_check(viennacl::matrix<double> &A)
{
    int n = A.size1() / 2;

//    std::unique_ptr<double> L(new double[n*n]);
    double *L = (double *) calloc(n * n, sizeof(double));
    if (L == nullptr) {
        throw std::runtime_error("Can not allocate memory for Cholesky matrix!");
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];

            double v = A(i, j) - s;

            if (i == j) {
                if (v <= 0) {
                    free(L);
                    throw std::runtime_error("Kernel Matrix is not semi-positive definite.");
                }
                L[i * n + j] = sqrt(v);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j]) * v;
            }
        }
    free(L);
}


void
svm_batch::solve_epsilon_svr_warm_start(
        const viennacl::matrix<double> &q_matrix,
        const vektor<double> &y,
        vektor<double> &alpha,
        double &in_out_rho,
        OnlineSVR &model)
{
    double smo_epsilon_divisor = model.get_smo_epsilon_divisor();
    double smo_cost_divisor = model.get_smo_cost_divisor();

    LOG4_DEBUG("R matrix dimensions " << model.get_r()->get_length_rows() << "x" << model.get_r()->get_length_cols());

    const double svr_c = model.get_svr_parameters().get_svr_C();
    const double svr_epsilon = model.get_svr_parameters().get_svr_epsilon();
    const ssize_t l = y.size();

    // Weights warm-start
    // EXPERIMENTAL: if weight KKT conditions do not hold,
    // force them to hold (at possibly extra expense to eps-tube violations)
    // This is because SMO seems to fail when the initial KKT are broken.
    vektor<double> alpha_vec(0., l);
    double weights_sum = 0;
#pragma omp parallel for reduction(+:weights_sum)
    for (ssize_t i = 0; i < l; ++i) {
        auto weight = alpha[i];
        if (weight > svr_c) weight = svr_c;
        else if (weight < -svr_c) weight = -svr_c;
        alpha_vec[i] = weight;
        weights_sum += weight;
    }

    // Rebalance the weight sum if needed
    if (std::abs(weights_sum) > svr_c / smo_cost_divisor) {
        LOG4_DEBUG("Stabilizing weights sum.");
        double approximate_number_vectors_same_sign = (double) l / 2.;
        //we expect this number of vectors of the same_sign, but we are not sure
        for (ssize_t i = 0; i < l; ++i) {
            double &weight = alpha_vec[i];
            LOG4_DEBUG("Iteration " << i << ", weights sum: " << std::abs(weights_sum) << ", weight: " << weight << " / " << svr_c);
            if (weights_sum < 0.) {
                // how much the weight can increase up to cost
                if ((weight > svr_epsilon / smo_epsilon_divisor) &&
                    (weight < svr_c)) { // the second condition is always true due to the cycle above
                    //we found a vector that we need.
                    approximate_number_vectors_same_sign = approximate_number_vectors_same_sign - .5 + 1.;
                    double wiggle_room = svr_c - weight;//positive
                    double partial_change = -weights_sum / approximate_number_vectors_same_sign;//positive
                    if (partial_change < wiggle_room) {
                        weight += partial_change;
                        weights_sum += partial_change;
                    } else {
                        weight += wiggle_room;
                        weights_sum += wiggle_room;
                    }
                } else {
                    // we didn't find a vector of the same sign.
                    approximate_number_vectors_same_sign = approximate_number_vectors_same_sign - .5;
                }
            } else {  // weights_sum > 0.

                if ((weight > -svr_c) && (weight < -svr_epsilon /
                                                           smo_epsilon_divisor)) { //the first condition is always true, due to the cycle above
                    double wiggle_room = svr_c + weight;
                    approximate_number_vectors_same_sign = approximate_number_vectors_same_sign - .5 + 1.;
                    LOG4_DEBUG("Wiggle room " << wiggle_room);
                    double partial_change = weights_sum / approximate_number_vectors_same_sign;
                    if (partial_change < wiggle_room) {
                        weight -= partial_change;
                        weights_sum -= partial_change;
                    } else {
                        weight -= wiggle_room;
                        weights_sum -= wiggle_room;
                    }
                } else {
                    // we didn't find a vector of the same sign.
                    approximate_number_vectors_same_sign = approximate_number_vectors_same_sign - .5;
                }
            }
        }
    }

#ifndef NDEBUG
    {
        if (std::abs(weights_sum) > svr_c / smo_cost_divisor)
            LOG4_ERROR(
                    "After stabilizing weight KKTs, still there are problems! Weights sum: "
                            << std::abs(weights_sum) <<
                            "/" << svr_c /

                            smo_cost_divisor);
        const auto weights_max_abs = alpha_vec.max_abs();
        if (weights_max_abs > svr_c)
            LOG4_ERROR(
                    "After stabilizing weight KKTs, still there are problems! Weights max abs " << weights_max_abs << " while cost is " << svr_c << ", deviating " << weights_max_abs - svr_c << " of KKT.");
    }
#endif

    std::vector<double> alpha2(2 * l, 0.);
    __omp_pfor_i (0, l,
                  const auto weight = alpha_vec[i];
        if (weight > 0.) alpha2[i] = weight;
        else if (weight < 0.) alpha2[l + i] = -weight;
    )

    // Linear term left seems to be how lower the sample is from the upper epsilon margin.
    // If the sample is higher than the upper tube margin, the linear term is negative, and if it is below the upper tube margin, it is positive.

    // Linear term right seems to be how higher the sample is from the lower epsilon margin.
    // The opposite rule applies:
    // If the sample is higher than the lower tube margin, the linear term is positive, and if it is below the upper tube margin, it is negative.

    // Logic then indicates that if both linear terms are >0, we are inside the eps-tube.
    // If one linear term is 0, then the other is positive, and we are on a margin.
    // If we have a negative value, then we are outside the eps-tube.

    // So this linear term is most likely directly related to the slack variables (cost function).

    vektor<double> linear_term(0., 2 * l);
    __omp_pfor_i(0, l,
        // For the upper margin
        linear_term[i] = svr_epsilon - y[i];
        // For the lower margin
        linear_term[i + l] = y[i] + svr_epsilon;
    )

    // This y_ vector relates to the target slack variables of SVM, the first half
    // to the positive weights, and the second half to the negative ones. Hence the -1 and 1
    std::vector<int> y_(2 * l, 1);
    __omp_pfor_i (l, y_.size(), y_[i] = -1)

    svr::smo::smo_solver smo(max_smo_iterations_, smo_epsilon_divisor_);
    const auto iter_num = smo.solve(
            q_matrix, svr::datamodel::diag(q_matrix), *linear_term.vcl(), y_, model.get_svr_parameters(), alpha2,
            in_out_rho);
    LOG4_INFO("SMO iterations count " << iter_num);

    __omp_pfor_i (0, l, alpha[i] = alpha2[i] - alpha2[l + i]);
    auto p_abs_alpha = alpha.abs_list();
    std::vector<int64_t> sv_indices;
    model.set_weights(alpha.clone());
    model.clear_indices();
    {
        size_t j = 0;
	//cannot be /* cilk_ */ for here
        for (size_t i = 0; i < (size_t) alpha.size(); ++i) {
            const auto abs_alpha_i = p_abs_alpha->get_value(i);
            if (svr::common::greater((double)abs_alpha_i,0.) && svr::common::less(abs_alpha_i, svr_c)) {
                model.add_to_support_set(i);
                sv_indices.push_back(i);
                j++;
            } else if (svr::common::is_zero((double)abs_alpha_i, (double)std::numeric_limits<float>::epsilon())) {
                model.add_to_remaining_set(i);
            } else if (svr::common::equal_to(abs_alpha_i, svr_c, (double)std::numeric_limits<float>::epsilon())) {
                model.add_to_error_set(i);
            } else
                LOG4_DEBUG("fabs_decision_func[i] > Cost [" << abs_alpha_i << " > " <<
                                                            svr_c << "], difference: " << abs_alpha_i - svr_c);
        }
        LOG4_INFO("SV #" << j);
    }
    delete p_abs_alpha;
    model.set_bias(-in_out_rho); // It's negative

    model.build_kernel_matrix();

    r_matrix r_matrix(model.get_kernel_matrix(), sv_indices, update_r_matrix_);
    model.set_R_matrix(r_matrix.get_r());
    LOG4_DEBUG("afterR matrix dims 1: " << model.get_r()->get_length_rows() << "x"
                                        << model.get_r()->get_length_cols());

#if DO_IN_PLACE_TRAINING_VALIDATION_SANITY_CHECK
    if (! model.VerifyKKTConditions())
        {
            LOG4_ERROR("KKT conditions for online -> batch train are not satisfied!");
        }
        else
        {
            LOG4_DEBUG("KKT conditions for online -> batch train are satisfied.");
        }
#endif //DO_IN_PLACE_TRAINING_VALIDATION_SANITY_CHECK

    if (model.get_error_set_elements_number() + model.get_support_set_elements_number() +
        model.get_remaining_set_elements_number() != l) {
        LOG4_ERROR("Sum of vectors in the three sets does not match reference data length: " << l);
    }

    LOG4_DEBUG("Batch train done, ErrorSet: " << model.get_error_set_elements_number());
    LOG4_DEBUG("afterR matrix dims 2: " << model.get_r()->get_length_rows() << "x"
                                        << model.get_r()->get_length_cols());

}


size_t
svm_batch::solve_epsilon_svr(
        const viennacl::matrix<double> &q_matrix,
        const viennacl::vector<double> &y,
        const SVRParameters &param,
        viennacl::vector<double> &alpha,
        double &out_rho,
        int loo)
{
    const auto l = y.size();
    std::vector<double> alpha2(2 * l, 0);
    viennacl::vector<double> linear_term = viennacl::scalar_vector<double>(2 * l, param.get_svr_epsilon());

    //extract subvectors, operate over them
    viennacl::range linear_term_range_left(0, l);
    viennacl::range linear_term_range_right(l, 2 * l);

    viennacl::vector_range<viennacl::vector<double>> linear_term_subvec_left(
            linear_term, linear_term_range_left);
    viennacl::vector_range<viennacl::vector<double>> linear_term_subvec_right(
            linear_term, linear_term_range_right);
    linear_term_subvec_left -= y;
    linear_term_subvec_right += y;

    if (loo != -1) // TODO Why LOO in Epsilon SVR?
    {
        const auto p_alpha = VCL_DOUBLE_PTR(alpha);
        __omp_pfor_i (0, l,
                      if (p_alpha[i] > 0.) alpha2[i] = p_alpha[i];
            if (p_alpha[i] < 0.) alpha2[i + l] = -p_alpha[i];
        )
        linear_term[loo] = linear_term[loo + l] = INF;
    }

    std::vector<int> y_(2 * l, 1);
    for (auto iter = y_.begin() + l; iter != y_.end(); ++iter) *iter = -1;

    svr::smo::smo_solver smo(max_smo_iterations_, smo_epsilon_divisor_);
    const auto iter_num = smo.solve(q_matrix, svr::datamodel::diag(q_matrix), linear_term, y_, param, alpha2,
                                    out_rho);

    {
        auto p_alpha = VCL_DOUBLE_PTR(alpha);
        __omp_pfor_i (0, l, p_alpha[i] = alpha2[i] - alpha2[l + i])
    }
    return iter_num;
}


void svm_batch::train(
        const arma::mat &learning_data,
        const arma::colvec &reference_data,
        OnlineSVR &model,
        const bool only_sv)
{
    LOG4_BEGIN();

    LOG4_DEBUG(
            "Training on " << learning_data.n_cols << " columns and " << learning_data.n_rows
                           << " rows. Only SV saved?: " << only_sv);
#ifdef CUDA_SMO
    smo::p_kernel_matrices quadratic_matrix(learning_data, model.get_svr_parameters());
#else
    smo::smo_quadratic_matrix quadratic_matrix(smo::kernel_matrix(learning_data, model.get_svr_parameters()));
#endif
    viennacl::vector<double> decision_func = viennacl::scalar_vector<double>(reference_data.n_rows, 0.);
    const auto n_instances = reference_data.n_rows;
    SyncArray<float_type> f_val(n_instances);
    SyncArray<int> y(n_instances);
    float_type *f_val_data = f_val.host_data();
    int *y_data = y.host_data();
    __omp_pfor_i (0, n_instances,
        f_val_data[i] = model.get_svr_parameters().get_svr_epsilon() - reference_data[i];
        y_data[i] = +1;
        f_val_data[i + n_instances] = - model.get_svr_parameters().get_svr_epsilon() - reference_data[i];
        y_data[i + n_instances] = -1;
    )

    SyncArray<float_type> alpha_2(n_instances * 2);
    double rho = 0;
#ifdef CUDA_SMO
    CSMOSolver solver;
    solver.solve(quadratic_matrix, y, alpha_2, rho, f_val,
            model.get_svr_parameters().get_svr_epsilon(), //PROPS.get_smo_epsilon_divisor(),
            model.get_svr_parameters().get_svr_C(),
            model.get_svr_parameters().get_svr_C(), DEFAULT_WORKING_SET, -1);
#else

    const auto iter_num = solve_epsilon_svr(
            quadratic_matrix, svr::common::arma_to_vcl(reference_data), model.get_svr_parameters(), decision_func, rho, -1);
    LOG4_INFO("SMO iterations count " << iter_num);
#endif

    // Sanity check
    if (decision_func.size() != decltype(decision_func.size())(reference_data.size()))
        LOG4_THROW("Decision function size " << decision_func.size() << " does not equal number of training rows " << reference_data.size());

#ifdef CUDA_SMO
    const float_type *alpha_2_data = alpha_2.host_data();
    viennacl::vector<double> fabs_decision_func(n_instances);
    double *fabs_decision_func_data = reinterpret_cast<double *>(fabs_decision_func.handle().ram_handle().get());
    __omp_pfor_i(0, n_instances,
        fabs_decision_func_data[i] = std::abs(alpha_2_data[i] - alpha_2_data[i + n_instances]);
    )
#else
     viennacl::vector<double> fabs_decision_func = viennacl::linalg::element_fabs(decision_func);
#endif

    //get SV, fill into onlineSVR model
    auto p_features = new vmatrix<double>();
    auto p_labels = new vektor<double>();

    std::vector<int64_t> sv_indices;
    size_t support_set_ix = 0;
    if (only_sv) {
        for (decltype(fabs_decision_func.size()) i = 0; i < fabs_decision_func.size(); ++i) {
            if (svr::common::is_zero((double) fabs_decision_func[i])) continue;
            p_features->add_row_copy(learning_data.row(i)); // TODO Reference not copy
            p_labels->add(reference_data(i));
            model.add_to_support_set(support_set_ix);
            sv_indices.push_back(support_set_ix);
#ifdef CUDA_SMO
            model.add_weight(alpha_2_data[i] - alpha_2_data[i + n_instances]);
#else
            model.add_weight(decision_func[i]);
#endif
            ++support_set_ix;
        }
    } else {
        for (decltype(fabs_decision_func.size()) i = 0; i < fabs_decision_func.size(); ++i) {
            // Add the feature vector, label, and weight regardless of the set it belongs to
            p_features->add_row_copy(learning_data.row(i));
            p_labels->add(reference_data(i));
#ifdef CUDA_SMO
            model.add_weight(alpha_2_data[i] - alpha_2_data[i + n_instances]);
#else
            model.add_weight(decision_func[i]);
#endif
            if (fabs_decision_func[i] > 0. &&
                fabs_decision_func[i] < model.get_svr_parameters().get_svr_C()) {
                model.add_to_support_set(i);
                sv_indices.push_back(i);
                ++support_set_ix;
            } else if (fabs_decision_func[i] == 0)
                model.add_to_remaining_set(i);
            else
                if (svr::common::equal_to(
                    (double) fabs_decision_func[i],
                    model.get_svr_parameters().get_svr_C(),
                    (double) std::numeric_limits<float>::epsilon()))
                model.add_to_error_set(i);
            else
                LOG4_THROW("Decision element " << i << " " << fabs_decision_func[i] << " bigger than cost " << model.get_svr_parameters().get_svr_C());
        }
    }

    model.set_samples_trained_number((size_t) p_labels->size());
    model.set_y(p_labels);
    model.set_x(p_features);
    model.set_bias(rho * viennacl::scalar<double>(-1.)); // It's negative

    LOG4_DEBUG("Features " << p_features->get_length_cols() << ", original learning data columns "
                           << learning_data.n_cols);


    if (!only_sv) {
        if (update_r_matrix_) {//if we are in paramtune, do not update and do not build kernel matrix!
            model.set_save_kernel_matrix(true);
            model.build_kernel_matrix();
    	    r_matrix r_matrix(model.get_kernel_matrix(), sv_indices, update_r_matrix_);
            model.set_R_matrix(r_matrix.get_r());
        } else {
            LOG4_DEBUG("We do not need r_matrix");
        }
    }


    if (model.get_error_set_elements_number() + model.get_support_set_elements_number() +
        model.get_remaining_set_elements_number() != (ssize_t) reference_data.size())
        LOG4_THROW("Sum of vectors in the three sets does not match reference data length " << reference_data.size());

    LOG4_DEBUG(
            "Batch train done, error set size " << model.get_error_set_elements_number() <<
                                                ", support set " << model.get_support_set_elements_number() <<
                                                ", remaining set " << model.get_remaining_set_elements_number()
    );

    LOG4_END();
}

} //batch
} //svr
