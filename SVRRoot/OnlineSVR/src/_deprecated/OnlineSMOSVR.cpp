#include "_deprecated/OnlineSMOSVR.tcc"
#include "_deprecated/smo.h"


namespace svr {

// Initialization
OnlineSVR::OnlineSVR()
{
    svr_parameters = SVRParameters();
    X = new vmatrix<double>();
    Y = new vektor<double>();
    p_weights = new vektor<double>();
    bias = .0;
    samples_trained_number = 0;
    p_support_set_indexes = new vektor<ssize_t>();
    p_error_set_indexes = new vektor<ssize_t>();
    p_remaining_set_indexes = new vektor<ssize_t>();
    p_r_matrix = new vmatrix<double>();
    p_kernel_matrix = new vmatrix<double>();
}

// Initialization
OnlineSVR::OnlineSVR(const SVRParameters &svr_parameters_) :
        svr_parameters(svr_parameters_)
{
    X = new vmatrix<double>();
    Y = new vektor<double>();
    p_weights = new vektor<double>();
    bias = .0;
    samples_trained_number = 0;
    p_support_set_indexes = new vektor<ssize_t>();
    p_error_set_indexes = new vektor<ssize_t>();
    p_remaining_set_indexes = new vektor<ssize_t>();
    p_r_matrix = new vmatrix<double>();
    p_kernel_matrix = new vmatrix<double>();

}

OnlineSVR::OnlineSVR(std::stringstream &input_stream) : OnlineSVR()
{
    if (!load_online_svr(*this, input_stream))
        throw std::runtime_error("Failed loading model from stream.");


}

OnlineSVR::~OnlineSVR()
{
    delete X;
    delete Y;
    delete p_weights;
    delete p_support_set_indexes;
    delete p_error_set_indexes;
    delete p_remaining_set_indexes;
    delete p_r_matrix;
    delete p_kernel_matrix;
}

void OnlineSVR::clear()
{
    samples_trained_number = 0;
    bias = 0;
    X->clear();
    Y->clear();
    p_weights->clear();
    p_support_set_indexes->clear();
    p_error_set_indexes->clear();
    p_remaining_set_indexes->clear();
    p_r_matrix->clear();
    p_kernel_matrix->clear();
    in_sample_margins.clear();
    save_kernel_matrix = false;
}



// Works for single values only, we assume data is normalized in the -1 .. 1 range
const double
OnlineSVR::calc_kernel_matrix_error(
        const arma::mat &y_train,
        const svr::smo::kernel_matrix &kernel_matrix)
{
    /*cilk::reducer<cilk::op_add<double>>*/ double red_err = 0;
    /*cilk::reducer<cilk::op_add<double>>*/ double red_ct = 0;
    /*cilk_*/for (size_t i = 0; i < kernel_matrix.size1(); ++i) {
        double row_err = 0;
        for (size_t j = 0; j <= i; ++j) {
            row_err += std::fmod(std::abs(y_train(i, 0) - y_train(j, 0)), kernel_matrix(i , j));
        }
        red_err += row_err;
        red_ct += i + 1;
    }
    return red_err / red_ct;
}


double
OnlineSVR::produce_kernel_matrices_variance(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train,
        const bool save_matrices)
{
    viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);
    svr::smo::kernel_matrix kernel_matrix(x_train, svr_parameters);
    const auto err = calc_kernel_matrix_error(y_train, kernel_matrix);
    const auto overall_score = err;
    LOG4_DEBUG("Error: " << err << ", overall score: " << overall_score << ", gamma: " <<
                svr_parameters.get_svr_kernel_param() << ", lambda: " << svr_parameters.get_svr_kernel_param2());
    return overall_score;
}


OnlineSVR &OnlineSVR::operator=(OnlineSVR &&other) noexcept
{
    if (this == &other) return *this;

    svr_parameters = other.svr_parameters;
    stabilized_learning = other.stabilized_learning;
    save_kernel_matrix = other.save_kernel_matrix;
    delete X;
    delete Y;
    X = other.X;
    Y = other.Y;
    other.X = nullptr;
    other.Y = nullptr;
    delete p_weights;
    p_weights = other.p_weights;
    other.p_weights = nullptr;
    bias = other.bias;
    delete p_support_set_indexes;
    p_support_set_indexes = other.p_support_set_indexes;
    other.p_support_set_indexes = nullptr;
    delete p_error_set_indexes;
    p_error_set_indexes = other.p_error_set_indexes;
    other.p_error_set_indexes = nullptr;
    delete p_remaining_set_indexes;
    p_remaining_set_indexes = other.p_remaining_set_indexes;
    other.p_remaining_set_indexes = nullptr;
    delete p_r_matrix;
    p_r_matrix = other.p_r_matrix;
    other.p_r_matrix = nullptr;
    delete p_kernel_matrix;
    p_kernel_matrix = other.p_kernel_matrix;
    other.p_kernel_matrix = nullptr;
    samples_trained_number = other.samples_trained_number;
    // TODO change copy to move
    in_sample_margins = other.in_sample_margins;
    return *this;
}

OnlineSVR *OnlineSVR::clone() const
{
    auto p_onlinesvr = new OnlineSVR(svr_parameters);
    //p_onlinesvr->svr_parameters = svr_parameters;
    p_onlinesvr->stabilized_learning = stabilized_learning;
    p_onlinesvr->save_kernel_matrix = save_kernel_matrix;
    p_onlinesvr->X = X->clone();
    p_onlinesvr->Y = Y->clone();
    p_onlinesvr->p_weights = p_weights->clone();
    p_onlinesvr->bias = bias;
    p_onlinesvr->p_support_set_indexes = p_support_set_indexes->clone();
    p_onlinesvr->p_error_set_indexes = p_error_set_indexes->clone();
    p_onlinesvr->p_remaining_set_indexes = p_remaining_set_indexes->clone();
    p_onlinesvr->p_r_matrix = p_r_matrix->clone();
    p_onlinesvr->p_kernel_matrix = p_kernel_matrix->clone();
    p_onlinesvr->samples_trained_number = samples_trained_number;
    p_onlinesvr->in_sample_margins = in_sample_margins;
    return p_onlinesvr;
}

// Attributes Operations
SVRParameters &OnlineSVR::get_svr_parameters()
{
    return svr_parameters;
}

void OnlineSVR::set_svr_parameters(const SVRParameters &svr_parameters_)
{
    svr_parameters = svr_parameters_;
}

ssize_t OnlineSVR::get_samples_trained_number() const
{
    return samples_trained_number;
}

ssize_t OnlineSVR::get_support_set_elements_number() const
{
    return p_support_set_indexes->size();
}

ssize_t OnlineSVR::get_error_set_elements_number() const
{
    return p_error_set_indexes->size();
}

ssize_t OnlineSVR::get_remaining_set_elements_number() const
{
    return p_remaining_set_indexes->size();
}

bool OnlineSVR::get_stabilized_learning() const
{
    return stabilized_learning;
}

void OnlineSVR::set_stabilized_learning(const bool stabilized_learning_arg)
{
    stabilized_learning = stabilized_learning_arg;
}

bool OnlineSVR::get_save_kernel_matrix() const
{
    return save_kernel_matrix;
}

void OnlineSVR::set_save_kernel_matrix(const bool save_kernel_matrix_arg)
{
    if (!save_kernel_matrix && save_kernel_matrix_arg)
        build_kernel_matrix();
    else if (!save_kernel_matrix_arg)
        p_kernel_matrix->clear();

    save_kernel_matrix = save_kernel_matrix_arg;
}

vektor<ssize_t> *OnlineSVR::get_support_set_indexes() const
{
    return p_support_set_indexes;
}

vektor<ssize_t> *OnlineSVR::get_error_set_indexes() const
{
    return p_error_set_indexes;
}

vektor<ssize_t> *OnlineSVR::get_remaining_set_indexes() const
{
    return p_remaining_set_indexes;
}

vektor<double> *OnlineSVR::get_weights() const
{
    return p_weights;
}

vmatrix<double> *OnlineSVR::get_x() const
{
    return X;
}

vmatrix<double> *OnlineSVR::get_r() const
{
    return p_r_matrix;
}

vektor<double> *OnlineSVR::get_y() const
{
    return Y;
}

vektor<double> &OnlineSVR::get_in_sample_margins()
{
    return in_sample_margins;
}

vmatrix<double> *OnlineSVR::get_kernel_matrix() const
{
    return p_kernel_matrix;
}

void OnlineSVR::set_samples_trained_number(const size_t samples_num)
{
    samples_trained_number = samples_num;
}

void OnlineSVR::clear_indices()
{
    p_support_set_indexes->clear();
    p_remaining_set_indexes->clear();
    p_error_set_indexes->clear();
}

void OnlineSVR::add_to_support_set(const ssize_t index)
{
    p_support_set_indexes->add(index);
}

void OnlineSVR::add_to_remaining_set(const ssize_t index)
{
    p_remaining_set_indexes->add(index);
}

void OnlineSVR::add_to_error_set(const ssize_t index)
{
    p_error_set_indexes->add(index);
}

void OnlineSVR::set_bias(const double bias)
{
    this->bias = bias;
}

void OnlineSVR::add_weight(const double weight)
{
    p_weights->add(weight);
}

void OnlineSVR::set_kernel_matrix(vmatrix<double> *matrix)
{
    delete p_kernel_matrix;
    p_kernel_matrix = matrix;
}

void OnlineSVR::set_weights(vektor<double> *weights)
{
    delete p_weights;
    p_weights = weights;
}

void OnlineSVR::set_x(vmatrix<double> *x)
{
    delete X;
    X = x;
}

void OnlineSVR::set_y(vektor<double> *y)
{
    delete Y;
    Y = y;
}

void OnlineSVR::clear_R_matrix()
{
    delete p_r_matrix;
}

void OnlineSVR::set_R_matrix(vmatrix<double> *r_matrix)
{
    p_r_matrix = r_matrix;
}


// Predict/Margin Operations
double OnlineSVR::predict(const ssize_t index) const // TODO Parallelize
{
#if 0
    if (!save_kernel_matrix) throw std::logic_error("Kernel matrix not saved.");
    cilk::reducer<cilk::op_add<double>> red_predicted_value(0.);
    /*cilk_*/for (decltype(samples_trained_number) i = 0; i < samples_trained_number; ++i)*red_predicted_value +=
                                                                                              p_weights->get_value(i) *
                                                                                              p_kernel_matrix->get_value(
                                                                                                      i, index);
    return red_predicted_value.get_value() + bias;
#endif
    return 0;
}

//std::mutex OnlineSVR::gpu_handler_mutex;
//svr::common::gpu_context OnlineSVR::static_paramtune_context;

double OnlineSVR::predict(const vektor<double> &features) const
{
    auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
    //vektor<double> kernel_values;

#ifdef VIENNACL_WITH_OPENCL
    auto context = svr::common::gpu_context();
    viennacl::ocl::context &ctx = context.ctx();
    vektor<double> kernel_values(0., X->get_length_rows());
    try {
        (*kernel)(ctx, features, *X, kernel_values);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Kernel predict failed. " << ex.what());
    }
#else
    (*kernel)(features, *X, kernel_values);
#endif
    return kernel_values.product_vector_scalar(*p_weights) + bias;
}


long double OnlineSVR::predict_stable(const vektor<double> &features)
{
    integrity_assured(true);
    return predict(features);
}

// TODO Port method to OpenCL kernel
vektor<double> OnlineSVR::predict(const vmatrix<double> &features) const
{
    LOG4_BEGIN();
    vektor<double> V(0., features.get_length_rows());
    if (this->X == &features && save_kernel_matrix)
        /*cilk_*/for (decltype(features.get_length_rows()) i = 0; i < features.get_length_rows(); ++i) V[i] = predict(i);
    else if (this->svr_parameters.get_kernel_type() == kernel_type_e::PATH) {
        // TODO Test, the OpenCL code seems like its freezing
        auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
        auto context = svr::common::gpu_context();
        viennacl::ocl::context &ctx = context.ctx();
        viennacl::matrix<double> vcl_kernel_matrix(this->X->get_length_rows(), features.get_length_rows(), ctx);
        viennacl::matrix<double> vcl_features(features.get_length_rows(), features.get_length_cols(), ctx);
        features.copy_to(vcl_features);
        viennacl::matrix<double> vcl_this_X(this->X->get_length_rows(), this->X->get_length_cols(), ctx);
        this->X->copy_to(vcl_this_X);
        try {
            (*kernel)(ctx, vcl_features, vcl_this_X, vcl_kernel_matrix);
        } catch (const std::exception &ex) {
            LOG4_ERROR("Kernel predict matrix failed. " << ex.what());
        }
        vmatrix<double> result_kernel_matrix(features.get_length_rows(), this->X->get_length_rows());
        result_kernel_matrix.copy_from(vcl_kernel_matrix);
        /*cilk_*/for (decltype(V.size()) i = 0;
                  i < result_kernel_matrix.get_length_rows(); ++i)V[i] = result_kernel_matrix.get_row_ref(
                    i).product_vector_scalar(*p_weights) + bias;
    } else { // Warning do not clog CILK thread pool with OpenCL tasks!
        for (decltype(features.get_length_rows()) i = 0; i < features.get_length_rows(); ++i)
            V[i] = predict(features.get_row_ref(i));
    }
    LOG4_END();
    return V;
}

vektor<double> OnlineSVR::predict_inplace_all() const
{
    vektor<double> V(0., X->get_length_rows());
    if (save_kernel_matrix)
        /*cilk_*/for (decltype(X->get_length_rows()) i = 0; i < X->get_length_rows(); ++i) V[i] = predict(i);
    else
        /*cilk_*/for (decltype(X->get_length_rows()) i = 0; i < X->get_length_rows(); ++i)V[i] = predict(X->get_row_ref(i));

    return V;
}

long double OnlineSVR::margin(const vektor<double> &X, const double Y) const
{
    return predict(X) - Y;
}


double OnlineSVR::margin(const size_t index) const
{
    auto predicted_label = save_kernel_matrix && index < (decltype(index)) p_kernel_matrix->get_length_cols() ?
                           predict(index) : predict(X->get_row_ref(index));
    if ((ssize_t) index >= Y->size())
        throw std::runtime_error(svr::common::formatter() << "Label index " << index << " does not exist!");
    return predicted_label - Y->get_value(index);
}

vektor<double> OnlineSVR::margin(const vmatrix<double> &X, const vektor<double> &Y)
{
    auto result_margin = predict(X);
    result_margin.subtract_vector(Y);
    return result_margin;
}

vektor<double> OnlineSVR::margin_inplace_all() const
{
    // TODO test if correct
    auto result_inplace_margin = predict_inplace_all();
    LOG4_DEBUG("Inplace predictions size " << result_inplace_margin.size() << ", labels count " << Y->size());
    result_inplace_margin.subtract_vector(Y);
    return result_inplace_margin;
}


// Other Kernel Operations
vmatrix<double> OnlineSVR::q(const vektor<ssize_t> &indexes1, const vektor<ssize_t> &indexes2) const // TODO Parallelize
{
    vmatrix<double> M(indexes1.size(), indexes2.size());
    if (!save_kernel_matrix) {
        auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
        __omp_pfor_i(0, indexes1.size(),
                     auto p_vec = new vektor<double>(0., indexes2.size());
            for (decltype(indexes2.size()) j = 0; j < indexes2.size(); ++j)
                (*p_vec)[j] = (*kernel)(X->get_row_ref(indexes1[i]), X->get_row_ref(indexes2[j]));
            M.set_row_ref_fast(p_vec, i);
        )
    } else
        __omp_pfor_i(0, indexes1.size(),
                     auto p_vec = new vektor<double>(0., indexes2.size());
            for (decltype(indexes2.size()) j = 0;
                      j < indexes2.size(); ++j)(*p_vec)[j] = p_kernel_matrix->get_value(indexes1[i], indexes2[j]);
            M.set_row_ref_fast(p_vec, i);
        )
    return M;
}


vmatrix<double> OnlineSVR::q(const vektor<ssize_t> &v) const // TODO Parallelize
{
    vmatrix<double> M(samples_trained_number, v.size());
    if (!save_kernel_matrix) {
        auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
        __omp_pfor_i (0, samples_trained_number,
                      auto p_vec2 = new vektor<double>(0., v.size());
            for (decltype(v.size()) j = 0; j < v.size(); ++j)
                (*p_vec2)[j] = (*kernel)(X->get_row_ref(i), X->get_row_ref(v[j])); //new factory call
            M.set_row_ref_fast(p_vec2, i);
        )
    } else
        __omp_pfor_i (0, samples_trained_number,
                      auto p_vec2 = new vektor<double>(0., v.size());
            for (decltype(v.size()) j = 0; j < v.size(); ++j)(*p_vec2)[j] = p_kernel_matrix->get_value(i, v[j]);
            M.set_row_ref_fast(p_vec2, i);
        )
    return M;
}


vektor<double> OnlineSVR::q(const vektor<ssize_t> &v, const ssize_t index) const // TODO Parallelize
{
    vektor<double> v2(0., v.size());
    if (!save_kernel_matrix) {
        auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
        __omp_pfor_i(0, v.size(), v2[i] = (*kernel)(X->get_row_ref(v[i]), X->get_row_ref(index)));
    } else
        __omp_pfor_i(0, v.size(), v2[i] = p_kernel_matrix->get_value(v.get_value(i), index));
    return v2;
}


vektor<double> OnlineSVR::q(const ssize_t index) const
{
    vektor<double> res(0., samples_trained_number);
    if (!save_kernel_matrix) {
        auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
        /* cilk_ */ for (decltype(samples_trained_number) i = 0; i < samples_trained_number; ++i)
            res[i] = (*kernel)(X->get_row_ref(i), X->get_row_ref(index));
    } else
        /* cilk_ */ for (decltype(samples_trained_number) i = 0;
                  i < samples_trained_number; ++i)res[i] = p_kernel_matrix->get_value(i, index);
    return res;
}


long double OnlineSVR::q(const ssize_t index1, const ssize_t index2) const
{
    return save_kernel_matrix ?
           p_kernel_matrix->get_value(index1, index2) :
           (*IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters))(X->get_row_ref(index1),
                                                                                            X->get_row_ref(index2));
}

// vmatrix R Operations
void OnlineSVR::add_to_r_matrix(
        const ssize_t sample_index,
        const set_type_e sample_old_set,
        const vektor<double> &beta,
        vektor<double> &gamma)
{
    if (p_r_matrix->get_length_rows() == 0) {
        auto p_v1 = new vektor<double>(2);
        p_v1->add(-q(sample_index, sample_index));
        p_v1->add(1);
        p_r_matrix->add_row_ref(p_v1);

        auto p_v2 = new vektor<double>(2);
        p_v2->add(1);
        p_v2->add(0);
        p_r_matrix->add_row_ref(p_v2);
    } else {
        vektor<double> *p_new_beta;
        if (sample_old_set == set_type_e::ERROR_SET || sample_old_set == set_type_e::REMAINING_SET) {
            // TODO We only need of beta and Gamma(SampleIndex)
            auto last_support = p_support_set_indexes->get_value(get_support_set_elements_number() - 1);
            p_support_set_indexes->remove_at(get_support_set_elements_number() - 1);
            p_new_beta = new vektor<double>{find_beta(sample_index)};
            gamma.set_at(sample_index, (double) find_gamma_sample(*p_new_beta, sample_index));
            p_support_set_indexes->add(last_support);
        } else
            p_new_beta = beta.clone();

        auto p_zeros = new vektor<double>(0., this->p_r_matrix->get_length_cols());
        p_r_matrix->add_col_copy(p_zeros);
        p_zeros->add(0.);
        p_r_matrix->add_row_ref(p_zeros);
        if (!svr::common::is_zero(gamma.get_value(sample_index))) {
            p_new_beta->add(1);
            double gamma_value = gamma.get_value(sample_index);
            size_t size_r_matrix = this->p_r_matrix->get_length_rows();
            /* cilk_ */ for (size_t i = 0; i < size_r_matrix; ++i) {
                /* cilk_ */ for (size_t j = 0; j < size_r_matrix; ++j) {
                    this->p_r_matrix->set_value(i, j, this->p_r_matrix->get_value(i, j) +
                                                      p_new_beta->get_value(i) * p_new_beta->get_value(j) /
                                                      gamma_value);
                }
            }
        }
        delete p_new_beta;
    }
}


void OnlineSVR::remove_from_r_matrix(const ssize_t sample_index)
{
    auto p_row = p_r_matrix->get_row_copy(sample_index + 1);
    p_row->remove_at(sample_index + 1);
    auto p_col = p_r_matrix->get_col_copy(sample_index + 1);
    p_col->remove_at(sample_index + 1);
    auto r_ii = p_r_matrix->get_value(sample_index + 1, sample_index + 1);
    p_r_matrix->remove_row(sample_index + 1);
    p_r_matrix->remove_col(sample_index + 1);
    if (!svr::common::is_zero(r_ii)) {
        /* auto p_r_variations = vmatrix<double>::product_vector_vector(p_col, p_row);
        p_r_variations->divide_scalar(r_ii);
        p_r_matrix->subtract_matrix(p_r_variations);
        delete p_r_variations;*/
        size_t size_r_matrix = p_row->size();
        /* cilk_ */ for (size_t i = 0; i < size_r_matrix; i++)/* cilk_ */ for (size_t j = 0;
                                                                 j < size_r_matrix; j++)this->p_r_matrix->set_value(
                        i, j, this->p_r_matrix->get_value(i, j) - (*p_col)[i] * (*p_row)[j] / r_ii);
    }
    delete p_row;
    delete p_col;
    if (p_r_matrix->get_length_rows() == 1) p_r_matrix->clear();
}


vektor<double> OnlineSVR::find_beta(const ssize_t sample_index) const
{
    auto qsi = q(*p_support_set_indexes, sample_index);
    qsi.add_at(1., 0);
    if (p_r_matrix->get_length_rows() < 1) LOG4_WARN("R matrix is empty!");
    vektor<double> beta = p_r_matrix->product_vector(qsi);
    beta.product_scalar(-1.);
    return beta;
}


vektor<double> OnlineSVR::find_gamma(const vektor<double> &beta, const ssize_t sample_index) const
{
    if (get_support_set_elements_number() == 0) return vektor<double>(1., samples_trained_number);
    auto qxi = q(sample_index);
    auto qxs = q(*p_support_set_indexes);
    vektor<double> ones(1., samples_trained_number);
    qxs.add_col_copy_at(ones, 0);
    auto gamma = qxs.product_vector(beta);
    gamma.sum_vector(qxi);
    return gamma;
}


long double OnlineSVR::find_gamma_sample(vektor<double> &beta, const ssize_t sample_index)
{
    if (get_support_set_elements_number() == 0) return 1;
    const auto qii = q(sample_index, sample_index);
    auto qsi = q(*p_support_set_indexes, sample_index);
    qsi.add_at(1., 0);
    return qii + beta.product_vector_scalar(qsi);
}


// KernelMatrix Operations
void OnlineSVR::add_to_kernel_matrix(const vektor<double> &features)
{
    const auto kernel_length_rows = p_kernel_matrix->get_length_rows();
    auto p_vec = new vektor<double>(0., kernel_length_rows);
    auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
    vektor<double> kernel_values;
    if (samples_trained_number > 1) {
#ifdef VIENNACL_WITH_OPENCL
        auto context = svr::common::gpu_context();
        auto &ctx = context.ctx();

        (*kernel)(ctx, features, *X, kernel_values);
#else
        (*kernel)(features, *X, kernel_values);

#endif
        for (ssize_t i = 0; i < kernel_length_rows; ++i)
            p_vec->set_at(i, kernel_values[i]);
        p_kernel_matrix->add_col_copy(p_vec); // p_vec is used below again
    }
    p_vec->add((*kernel)(features, features));
    p_kernel_matrix->add_row_ref(p_vec);
}

void OnlineSVR::remove_from_kernel_matrix(const ssize_t sample_index)
{
    if (p_kernel_matrix->get_length_rows() > 1) {
        p_kernel_matrix->remove_row(sample_index);
        p_kernel_matrix->remove_col(sample_index);
    } else {
        p_kernel_matrix->remove_row(sample_index);
    }
}


void OnlineSVR::sanity_ensured()
{
    if (X->get_length_rows() != samples_trained_number ||
        Y->size() != samples_trained_number)
        throw std::logic_error(
                svr::common::formatter() <<
                                         "Model is corrupt, samples trained " << samples_trained_number <<
                                         " is different than labels count " << X->get_length_rows() << " ");
}


void OnlineSVR::build_kernel_matrix()
{
#ifdef VIENNACL_WITH_OPENCL
    auto context = svr::common::gpu_context();
    auto &ctx = context.ctx();
    viennacl::matrix<double> support_vectors_data_vcl(
            X->get_length_rows(), X->get_length_cols(), ctx);
    X->copy_to(support_vectors_data_vcl);
    viennacl::matrix<double> dense_kernel_matrix(X->get_length_rows(), X->get_length_rows(), ctx);
    (*IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters))(ctx, support_vectors_data_vcl,
                                                                                     dense_kernel_matrix);
    viennacl::matrix<double> kernel_matrix_cpu(X->get_length_rows(), X->get_length_rows(),
                                               viennacl::context(viennacl::MAIN_MEMORY));
    viennacl::fast_copy(dense_kernel_matrix, (double *) kernel_matrix_cpu.handle().ram_handle().get());
    p_kernel_matrix->clear();
    p_kernel_matrix->copy_from(kernel_matrix_cpu);
    sanity_ensured();
#else
    if (p_kernel_matrix->get_length_rows() != samples_trained_number) {
        p_kernel_matrix->clear();
        sanity_ensured();
        for (ssize_t i = 0; i < samples_trained_number; ++i)
            add_to_kernel_matrix(X->get_row_ref(i)); // can't be parallelized now
    } else {
        auto kernel = IKernel<double>::get_kernel(p_svr_parameters.get_kernel_type(),p_svr_parameters);
        for (ssize_t i = 0; i < samples_trained_number; ++i) {
            for (ssize_t j = 0; j <= i; ++j) {
                const auto value = (*kernel)(X->get_row_ref(i), X->get_row_ref(j));
                p_kernel_matrix->set_value(i, j, value);
                p_kernel_matrix->set_value(j, i, value);
            }
        }
    }
#endif
    save_kernel_matrix = true;
}


bool OnlineSVR::operator==(OnlineSVR const &o) const
{
    return
            samples_trained_number == o.samples_trained_number
            && stabilized_learning == o.stabilized_learning
            && save_kernel_matrix == o.save_kernel_matrix

            && X && o.X && *X == *o.X
            && Y && o.Y && *Y == *o.Y
            && p_weights && o.p_weights && *p_weights == *o.p_weights
            && bias == o.bias

            && p_support_set_indexes && o.p_support_set_indexes && *p_support_set_indexes == *o.p_support_set_indexes
            && p_error_set_indexes && o.p_error_set_indexes && *p_error_set_indexes == *o.p_error_set_indexes
            && p_remaining_set_indexes && o.p_remaining_set_indexes &&
            *p_remaining_set_indexes == *o.p_remaining_set_indexes
            && p_r_matrix && o.p_r_matrix && *p_r_matrix == *o.p_r_matrix
            && p_kernel_matrix && o.p_kernel_matrix && *p_kernel_matrix == *o.p_kernel_matrix

            && svr_parameters.get_svr_C() == o.svr_parameters.get_svr_C()
            && svr_parameters.get_svr_epsilon() == svr_parameters.get_svr_epsilon()
            && svr_parameters.get_kernel_type() == svr_parameters.get_kernel_type()
            && svr_parameters.get_svr_kernel_param() == svr_parameters.get_svr_kernel_param()
            && svr_parameters.get_svr_kernel_param2() == svr_parameters.get_svr_kernel_param2();
}


size_t OnlineSVR::size_of() const
{
    size_t result = sizeof(OnlineSVR);
    return result;
}


} // svr
