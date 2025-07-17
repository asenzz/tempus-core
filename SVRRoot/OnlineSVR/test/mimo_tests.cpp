
// Created by sstoyanov on 10/31/18.
//

#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <cstdlib>
#include "../include/onlinesvr.hpp"
#include "test_harness.hpp"
#include "onlinesvr.hpp"
#include "util/math_utils.hpp"

#define DEFAULT_ITERS (2)
#define TOLERANCE 0.001



void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
#if 0
    MKL_INT n = a.n_rows;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = n;
    MKL_INT ldb = n;
    solved = b;
    std::vector<double> a_(a.n_elem);
    std::memcpy(a_.data(), a.memptr(), sizeof(double) * a.n_elem);
    //MKL_INT info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', n, nrhs, a_.data(), lda, solved.memptr(), ldb);
    //std::vector<double> a_(a.n_elem);
    //std::memcpy(a_.data(), a.memptr(), sizeof(double) * a.n_elem);
    std::vector<MKL_INT> ipiv(a.n_rows);
    MKL_INT info = LAPACKE_dgesv( LAPACK_COL_MAJOR, n, nrhs, a_.data(), lda, ipiv.data(), (solved.memptr()), ldb );
    if (info > 0) {
        LOG4_ERROR("The leading minor of order " << info
                                                 << " is not positive definite; The solution could not be computed.");
        throw std::runtime_error("Matrix not positive semi-definite");
    }
#endif
}

TEST(mimo_train_predict, batch_train)
{
    svr::datamodel::OnlineSVR online_svr_tmp;
    svr::datamodel::OnlineSVR_ptr mimo_model;

    try {
        mimo_model = std::make_shared<svr::datamodel::OnlineSVR>();// svr::OnlineMIMOSVR::load("../SVRRoot/OnlineSVR/test/test_data/10k_mimo_model_level1_close");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::datamodel::OnlineSVR online_svr_(0, 0, mimo_model->get_param_set());

    LOG4_DEBUG("multistep_len is " << mimo_model->get_multiout());

    arma::mat x_train = mimo_model->get_features();
    arma::mat y_train = mimo_model->get_labels();

    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    std::cout << "Learning data rows are " << x_train.n_rows << ", reference data rows " << y_train.n_rows;

    PROFILE_INFO(online_svr_.batch_train(std::make_shared<arma::mat>(x_train), std::make_shared<arma::mat>(y_train), {},
                                                  bpt::not_a_date_time), "Batch Train");

//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(1,2), mimo_model->get_weights(0)(1,2), TOLERANCE);
//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(2,3), mimo_model->get_weights(0)(2,3), TOLERANCE);
//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(3,5), mimo_model->get_weights(0)(3,5), TOLERANCE);

}


TEST(mimo_train_predict, chunk_train)
{
    svr::datamodel::OnlineSVR online_svr_tmp;
    svr::datamodel::OnlineSVR_ptr mimo_model;

    try {
        mimo_model = std::make_shared<svr::datamodel::OnlineSVR>();
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::datamodel::OnlineSVR online_svr_(0, 0, mimo_model->get_param_set());

    LOG4_DEBUG("multistep_len is " << mimo_model->get_multiout());

    arma::mat x_train = mimo_model->get_features();
    arma::mat y_train = mimo_model->get_labels();

    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    std::cout << "Learning data rows are " << x_train.n_rows << ", reference data rows " << y_train.n_rows;

    PROFILE_INFO(
            online_svr_.batch_train(std::make_shared<arma::mat>(x_train), std::make_shared<arma::mat>(y_train), {},
                        bpt::not_a_date_time), "Batch Train");

//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(1,2), mimo_model->get_weights(0)(1,2), TOLERANCE);
//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(2,3), mimo_model->get_weights(0)(2,3), TOLERANCE);
//    EXPECT_NEAR(online_svr_.get_mimo_model().get_weights(0)(3,5), mimo_model->get_weights(0)(3,5), TOLERANCE);

}
#if 0
TEST(mimo_train_predict, batch_train_tasks)
{
    svr::OnlineSVR online_svr_tmp;
    OnlineMIMOSVR_ptr mimo_model;

    try {
        mimo_model = online_svr_tmp.load("../SVRRoot/OnlineSVR/test/test_data/10k_mimo_model_level1_close");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
        return;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineSVR online_svr_(mimo_model->get_svr_parameters(), mimo_model->get_multistep_len());

    arma::mat x_train = mimo_model->get_learning_matrix();
    arma::mat y_train = mimo_model->get_reference_matrix();

    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    const auto train_tasks_str = getenv("TRAIN_TASKS");
    const int number_of_tasks = train_tasks_str ? std::atoi(train_tasks_str) : DEFAULT_ITERS;
//    const auto number_of_tasks = 16;
    std::vector<std::future<void>> batch_train_tasks(number_of_tasks);
    std::vector<svr::OnlineMIMOSVR> mimo_models;
    std::vector<OnlineSVR_ptr> osvr_models(number_of_tasks);
    for(int i = 0; i < number_of_tasks; ++i)
    {
        osvr_models[i] = std::make_shared<svr::OnlineSVR>(mimo_model->get_svr_parameters(),
                                                          mimo_model->get_multistep_len());
        auto & mimo_model_ = osvr_models[i]->get_mimo_model();
        batch_train_tasks[i] = std::async(std::launch::async, (void(svr::OnlineMIMOSVR::*)(const arma::mat &,const arma::mat &, const bool, const size_t))&svr::OnlineMIMOSVR::batch_train, &(mimo_model_), x_train, y_train, false, 1);
    }
    for(int i = 0; i < number_of_tasks; ++i) {
        batch_train_tasks[i].get();
        auto & mimo_model_ = osvr_models[i]->get_mimo_model();

        EXPECT_NEAR(mimo_model_.get_weights(0)(1, 2),
                    mimo_model->get_weights(0)(1, 2), TOLERANCE);
        EXPECT_NEAR(mimo_model_.get_weights(0)(2, 3),
                    mimo_model->get_weights(0)(2, 3), TOLERANCE);
        EXPECT_NEAR(mimo_model_.get_weights(0)(3, 5),
                    mimo_model->get_weights(0)(3, 5), TOLERANCE);

        LOG4_DEBUG("Model number " << i << " has finished.");
    }
}

TEST(mimo_train_predict, batch_train_repeated)
{
    svr::OnlineSVR online_svr_tmp;
    OnlineMIMOSVR_ptr mimo_model;

    try {
        mimo_model = online_svr_tmp.load("../SVRRoot/OnlineSVR/test/test_data/10k_mimo_model_level1_close");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
        return;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineSVR online_svr_(mimo_model->get_svr_parameters(), mimo_model->get_multistep_len());

    arma::mat x_train = mimo_model->get_learning_matrix();
    arma::mat y_train = mimo_model->get_reference_matrix();

    const auto iters_str = getenv("iters");
    const int iterations = iters_str ? atoi(iters_str) : DEFAULT_ITERS;

    LOG4_DEBUG("Iterations are" << iterations);


    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    auto & mimo_model_ = online_svr_.get_mimo_model();

    for(int i = 0; i < iterations; ++i)
    {
        svr::OnlineSVR online_svr_(mimo_model->get_svr_parameters(),
                                   mimo_model->get_multistep_len());

        PROFILE_INFO(mimo_model_.batch_train(x_train, y_train, false),
                          "Batch Train");
        LOG4_DEBUG("weights: " << mimo_model_.get_weights(0)(1,2) << " " << mimo_model_.get_weights(0)(2,3) << " " << mimo_model_.get_weights(0)(100,7));
        EXPECT_NEAR(mimo_model_.get_weights(0)(1, 2),
                    mimo_model->get_weights(0)(1, 2), TOLERANCE);
        EXPECT_NEAR(mimo_model_.get_weights(0)(2, 3),
                    mimo_model->get_weights(0)(2, 3), TOLERANCE);
        EXPECT_NEAR(mimo_model_.get_weights(0)(3, 5),
                    mimo_model->get_weights(0)(3, 5), TOLERANCE);

    }

}


TEST(mimo_train_predict, batch_train_gr)
{
    svr::OnlineSVR online_svr_tmp;
    OnlineMIMOSVR_ptr mimo_model;

    try {
        mimo_model = online_svr_tmp.load("../SVRRoot/OnlineSVR/test/test_data/5000_mimo_model.txt");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
        return;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineSVR online_svr_(mimo_model->get_svr_parameters(), mimo_model->get_multistep_len());

    arma::mat x_train = mimo_model->get_learning_matrix();
    arma::mat y_train = mimo_model->get_reference_matrix();

    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    std::cout << "Learning data rows are " << x_train.n_rows << ", reference data rows " << y_train.n_rows;

    PROFILE_INFO(online_svr_.get_mimo_model().nesterov_train(x_train, y_train, 100),
                      "Batch Train NADAM epoch count 100");

    //online_svr_.get_mimo_model().nesterov_train(x_train, y_train, 50);

//    ASSERT_FP_EQ(online_svr_.get_mimo_model().get_weights(1)(1,2), online_svr_tmp.get_mimo_model().get_weights(1)(1,2));
//    ASSERT_FP_EQ(online_svr_.get_mimo_model().get_weights(1)(2,3), online_svr_tmp.get_mimo_model().get_weights(1)(2,3));
//    ASSERT_FP_EQ(online_svr_.get_mimo_model().get_weights(1)(3,5), online_svr_tmp.get_mimo_model().get_weights(1)(3,5));
    //ASSERT_FP_EQ(online_svr_.get_mimo_model().get_weights(1)(4,2), online_svr_tmp.get_mimo_model().get_weights(1)(4,2));
    LOG4_DEBUG("chunk_weights are " << online_svr_.get_mimo_model().get_weights(0)(1,2) << " " << mimo_model->get_weights(0)(1,2));
    LOG4_DEBUG("chunk_weights are " << online_svr_.get_mimo_model().get_weights(0)(2,3) << " " << mimo_model->get_weights(0)(2,3));
    LOG4_DEBUG("chunk_weights are " << online_svr_.get_mimo_model().get_weights(0)(3,5) << " " << mimo_model->get_weights(0)(3,5));
    LOG4_DEBUG("chunk_weights are " << online_svr_.get_mimo_model().get_weights(0)(4,2) << " " << mimo_model->get_weights(0)(4,2));

   // svr::common::armd::print_mat(online_svr_.get_mimo_model().get_weights(1));

    online_svr_.save_onlinemimosvr("nesterov_train_test_mimo_model.txt");

}


arma::mat delete_from_r_matrix(arma::mat kernel, arma::mat r, size_t index)
{
    arma::uvec rows_and_cols = arma::linspace<arma::uvec>(0, r.n_rows - 1, r.n_rows);
    rows_and_cols.shed_row(index);
    arma::mat sub_matrix = r.submat(rows_and_cols, rows_and_cols);

    arma::mat no_row = r.rows(rows_and_cols);
    arma::mat no_col = r.cols(rows_and_cols);

    return sub_matrix - no_row.col(index) * (no_col.row(index) / r(index, index));
}

TEST(mimo_online_train, delete_from_r_matrix)
{
    arma::mat kk(3, 3);
    kk(0, 0) = 4;
    kk(0, 1) = -2;
    kk(0, 2) = 3;
    kk(1, 0) = 8;
    kk(1, 1) = -3;
    kk(1, 2) = 5;
    kk(2, 0) = 7;
    kk(2, 1) = -2;
    kk(2, 2) = 4;

    arma::mat new_km = kk;
    new_km.shed_row(0);
    new_km.shed_col(0);


    arma::mat real_r_matrix, r;
    arma::mat singular = arma::eye(new_km.n_rows, new_km.n_cols);
    do_mkl_solve(new_km, singular, real_r_matrix);
    singular = arma::eye(kk.n_rows, kk.n_cols);
    do_mkl_solve(kk, singular, r);

    LOG4_DEBUG("r sizes " << r.n_rows << " " << r.n_cols);

    arma::mat computed_r_matrix = delete_from_r_matrix(kk, r, 0);
    LOG4_DEBUG("some values are 0,1 1,2 2,1 " << real_r_matrix(0, 1) << " and " << computed_r_matrix(0, 1) << " " << real_r_matrix(1, 0) << " and " << computed_r_matrix(1, 0) << " " << real_r_matrix(1, 1) << " and " << computed_r_matrix(1, 1));

    std::stringstream ss;
    arma::umat are_equal = computed_r_matrix == real_r_matrix;
    for(size_t i = 0; i < are_equal.n_rows; ++i){
        for(size_t j = 0; j < are_equal.n_cols; ++j) {
            ASSERT_FP_EQ(real_r_matrix(i, j), computed_r_matrix(i, j));
            ss << are_equal(i, j) << " ";
        }
        ss << "\n";
    }

//    LOG4_DEBUG("some values are 2,4 5,6 8,3 " << real_r_matrix(2, 4) << "," << computed_r_matrix(2, 4) << " " << real_r_matrix(5, 6) << "," << computed_r_matrix(5, 6) << real_r_matrix(8, 3) << "," << computed_r_matrix(8, 3));
    LOG4_DEBUG("Matrix: " << ss.str());

}


void add_to_r_matrix(const arma::mat &p_kernel_matrices, arma::mat &r_matrix, size_t index, double correction)
{
    auto old_r_mat = r_matrix;
    size_t last_elem = p_kernel_matrices.n_rows - 1;
    arma::uvec others = arma::linspace<arma::uvec>(0, last_elem, p_kernel_matrices.n_rows);
    others.shed_row(index);
    arma::uvec element(1);
    element(0) = index;
    arma::mat a11 = p_kernel_matrices.submat(others, others) + arma::eye(last_elem, last_elem) * correction;
    arma::mat a12 = p_kernel_matrices.submat(others, element);
    arma::mat a21 = p_kernel_matrices.submat(element, others);

    const double a22 = p_kernel_matrices(index, index) + correction;
    const arma::mat f22 = a22 - a21 * r_matrix * a12;
    const arma::mat r_f22 = 1. / f22;

    const arma::mat prod_a11m1_a12 = r_matrix * a12;
    const arma::mat prod_a21_a11m1 = a21 * r_matrix;
    r_matrix = r_matrix + prod_a11m1_a12 * r_f22 * prod_a21_a11m1;
    const arma::mat new_a12 = (-r_matrix * a12) / a22;
    arma::mat new_a21 = -(a21 * r_matrix) / a22;
    r_matrix.insert_cols(index, new_a12);
    new_a21.insert_cols(index, r_f22);
    r_matrix.insert_rows(index, new_a21);


    arma::mat sing = arma::eye(p_kernel_matrices.n_rows, p_kernel_matrices.n_cols);
    arma::mat newmat = r_matrix * (p_kernel_matrices + sing * correction);
    arma::mat oldmat = old_r_mat * a11;
    double err1 = sqrt(arma::mean(
            arma::mean((oldmat - arma::eye(last_elem, last_elem)) % (oldmat - arma::eye(last_elem, last_elem)), 0)));
    double err2 = sqrt(arma::mean(arma::mean((newmat - sing) % (newmat - sing), 0)));
    LOG4_DEBUG("err1 err2 " << err1 << " " << err2);

}

#if 0
arma::mat add_to_r_matrix(arma::mat p_kernel_matrices, arma::mat r_matrix, size_t index, double correction)
{
    size_t last_elem = p_kernel_matrices.n_rows - 1;
    arma::uvec others = arma::linspace<arma::uvec>(0, last_elem, p_kernel_matrices.n_rows);
    others.shed_row(index);
    arma::uvec element(1);
    element(0) = index;
    arma::mat a11 = p_kernel_matrices.submat(others, others) + arma::eye(last_elem, last_elem) * correction;
    arma::mat a12 = p_kernel_matrices.submat(others, element);
    arma::mat a21 = p_kernel_matrices.submat(element, others);

    const double a22 = p_kernel_matrices(index, index) + correction;

    const double r_a22 = 1. / a22;

    const arma::mat f11 = r_matrix - a12 * r_a22 * a21;

    const arma::mat f22 = a22 - a21 * r_matrix * a12;
    const arma::mat r_f22 = 1. / f22;

    const arma::mat xx = r_matrix * a12;
    const arma::mat xx2 = a21 * r_matrix;
    r_matrix  = r_matrix + xx * r_f22 * xx2;

    arma::mat new_a12 = -r_matrix * a12 * r_a22;
    arma::mat new_a21 = -r_a22 * a21 * r_matrix;

    r_matrix.insert_cols(index, new_a12);
    new_a21.insert_cols(index, r_f22);
    r_matrix.insert_rows(index, new_a21);

    return r_matrix;
}

#endif


TEST(mimo_online_train, add_to_r_matrix)
{
    arma::mat k(2, 2);
    k(0, 0) = 4;
    k(0, 1) = -2;
    k(1, 0) = 8;
    k(1, 1) = -3;

    arma::mat r = arma::inv(k);

    arma::mat k2(3, 3);
    k2(0, 0) = 4;
    k2(0, 1) = -2;
    k2(0, 2) = 3;
    k2(1, 0) = 8;
    k2(1, 1) = -3;
    k2(1, 2) = 5;
    k2(2, 0) = 7;
    k2(2, 1) = -2;
    k2(2, 2) = 4;

    arma::mat real_r_matrix = arma::inv(k2);

    add_to_r_matrix(k2, r, 2, 0);
    svr::common::armd::print_mat(r);

    LOG4_DEBUG("some values are 1,1 1,2 2,0 " << real_r_matrix(1, 1) << "," << r(1, 2)
                                              << " " << real_r_matrix(1, 2) << "," << r(2, 0) << real_r_matrix(2, 0)
                                              << "," << r(2, 0));

    std::stringstream ss;
    arma::umat are_equal = r == real_r_matrix;
    for(size_t i = 0; i < are_equal.n_rows; ++i){
        for(size_t j = 0; j < are_equal.n_cols; ++j) {
            ASSERT_FP_EQ(real_r_matrix(i, j), r(i, j));
            ss << are_equal(i, j) << " ";
        }
        ss << "\n";
    }
    LOG4_DEBUG("Matrix: " << ss.str());

}

static double error_mult(const arma::mat & km, const arma::mat & rm){
    arma::mat sing=arma::eye(km.n_rows, km.n_cols);

    arma::mat new_mat = km*rm;
    double err = sqrt(arma::mean(arma::mean((new_mat-sing) % (new_mat-sing), 0)));
    arma::mat invkm;
    arma::mat save_km=km;
    do_mkl_solve(save_km, sing, invkm);
   
    LOG4_DEBUG("real error"<< sqrt(arma::mean(arma::mean((km*invkm-sing) % (km*invkm-sing), 0))));
    return err;
}




TEST(mimo_online_train, batch_train_forget_learn)
{
    svr::OnlineSVR online_svr_tmp;
    OnlineMIMOSVR_ptr mimo_model;

    try {
        mimo_model = online_svr_tmp.load("../SVRRoot/OnlineSVR/test/test_data/5000_mimo_model.txt");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
        return;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineSVR online_svr_(mimo_model->get_svr_parameters(), mimo_model->get_multistep_len());

    arma::mat x_train = mimo_model->get_learning_matrix();
    arma::mat y_train = mimo_model->get_reference_matrix();

    LOG4_DEBUG("Learning data rows " << x_train.n_rows
                                     << ", reference data rows " << y_train.n_rows);

    PROFILE_INFO(online_svr_.get_mimo_model().batch_train(x_train, y_train, true),
                      "Batch Train");

//    size_t last_row_ix = x_train.n_rows - 1;

    auto mimo_model_ = online_svr_.get_mimo_model();
    arma::mat k = mimo_model_.get_kernel_matrix();
    mimo_model_.init_r_matrix();
    arma::mat r = mimo_model_.get_r_matrix();
    LOG4_DEBUG("sizes of k are " << k.n_rows << " " << k.n_cols);
    LOG4_DEBUG("sizes of r_matrix are " << r.n_rows << " " << r.n_cols);

    const double correction = 1 / mimo_model_.get_svr_parameters().get_svr_C();
    LOG4_DEBUG("before forget " << error_mult(k + arma::eye(k.n_rows,k.n_cols) * correction, r));

    mimo_model_.forget(0, 0);


    LOG4_DEBUG("after forget");

    arma::mat new_km = k;
    arma::rowvec row0 = new_km.row(0);
    arma::colvec col0 = new_km.col(0);
    col0.shed_row(0);
    new_km.shed_row(0);
    new_km.shed_col(0);

    arma::mat computed_r_matrix = delete_from_r_matrix(k, r, 0);
//    LOG4_DEBUG("r_matrix values and computed r_matrix arma::inv are 0,1 5,22 54,8 " << real_r_matrix(0, 1) << " and " << computed_r_matrix(0, 1) << " " << real_r_matrix(5, 23) << " and " << computed_r_matrix(5, 23) << " " << real_r_matrix(54, 8) << " and " << computed_r_matrix(54, 8));

    arma::mat k_f = mimo_model_.get_kernel_matrix();
    arma::mat r_f = mimo_model_.get_r_matrix();


    LOG4_DEBUG("after forget " << error_mult(k_f + arma::eye(size(k_f,0),size(k_f,1)) * correction , r_f));

    LOG4_DEBUG("kernel values values are 0,1 5,22 54,8 " << k_f(0, 1) << " and " << new_km(0, 1) << " " << k_f(5, 23) << " and " << new_km(5, 23) << " " << k_f(54, 8) << " and " << new_km(54, 8));
    LOG4_DEBUG("kernel values values are 0,1 5,22 54,8 " << k_f(1, 0) << " and " << new_km(1, 0) << " " << k_f(23, 5) << " and " << new_km(23, 5) << " " << k_f(8, 54) << " and " << new_km(8, 54));
    LOG4_DEBUG("my functions comparison are 0,1 5,22 54,8 " << r_f(0, 1) << " and " << computed_r_matrix(0, 1) << " " << r_f(5, 23) << " and " << computed_r_matrix(5, 23) << " " << r_f(54, 8) << " and " << computed_r_matrix(54, 8));

    mimo_model_.learn(x_train.row(0), y_train.row(0));

    LOG4_DEBUG("after learn");

    new_km.insert_cols(new_km.n_cols, col0);
    const auto last_value = row0(0);
    LOG4_DEBUG("row0 size " << row0.n_cols << " " << row0.n_elem);
    row0.shed_col(0);
    LOG4_DEBUG("row0 after shed size " << row0.n_cols << " " << row0.n_elem << " insert at " << row0.n_cols);
    row0.insert_cols(row0.n_cols, 1);
    row0(row0.n_cols - 1) = last_value;
    LOG4_DEBUG("row0 after insert size " << row0.n_cols << " " << row0.n_elem << " insert at " << row0.n_cols);
    LOG4_DEBUG("newkm size size " << new_km.n_rows << " " << new_km.n_cols);
    new_km.insert_rows(new_km.n_rows, row0);
    add_to_r_matrix(new_km, computed_r_matrix, new_km.n_rows - 1, correction);
    arma::mat r_l2 = computed_r_matrix;
    arma::mat k_l = mimo_model_.get_kernel_matrix();
    arma::mat r_l = mimo_model_.get_r_matrix();
    LOG4_DEBUG("after learn " << error_mult(k_l + arma::eye(size(k_l,0),size(k_l,1)) * correction , r_l));

    LOG4_DEBUG("r_matrix values of learn are 0,1 5,22 54,8 " << r_l2(0, 1) << " and " << r_l(0, 1) << " " << r_l2(5, 23) << " and " << r_l(5, 23) << " " << r_l2(54, 8) << " and " << r_l(54, 8));



    arma::mat new_i = (k_l + arma::eye(size(k_l,0),size(k_l,1))*1/mimo_model_.get_svr_parameters().get_svr_C())  * r_l;
    arma::mat old_i = (k +  arma::eye(size(k,0),size(k,1))*1./mimo_model_.get_svr_parameters().get_svr_C())  * r;
    LOG4_DEBUG("singular matrix sizes are " << new_i.n_rows << " " << new_i.n_cols);
    LOG4_DEBUG("singular values values are 0,1 5,22 54,8 " << new_i(0, 1) << " and " << old_i(0, 1) << " " << new_i(5, 23) << " and " << old_i(5, 23) << " " << new_i(54, 8) << " and " << old_i(54, 8));
    arma::mat sing(new_i.n_rows, new_i.n_cols, arma::fill::eye);

    auto err = sqrt(arma::mean(arma::mean((new_i-sing) % (new_i-sing), 0)));
    LOG4_DEBUG("err is " << err);

    auto err_old = sqrt(arma::mean(arma::mean((old_i-sing) % (old_i-sing), 0)));
    LOG4_DEBUG("err_old is " << err_old);

    std::stringstream ss;
    arma::umat are_equal = r == r_l;
    for(size_t i = 0; i < are_equal.n_rows; ++i){
        for(size_t j = 0; j < are_equal.n_cols; ++j) {
            ASSERT_FP_EQ(r(i, j), r_l(i, j));
            ss << are_equal(i, j) << " ";
        }
        ss << "\n";
    }

//    LOG4_DEBUG("some values are 2,4 5,6 8,3 " << real_r_matrix(2, 4) << "," << computed_r_matrix(2, 4) << " " << real_r_matrix(5, 6) << "," << computed_r_matrix(5, 6) << real_r_matrix(8, 3) << "," << computed_r_matrix(8, 3));
    LOG4_DEBUG("Matrix: " << ss.str());

}



TEST(mimo_online_train, multiple_forget_learn)
{
    const size_t iters = 100;

    svr::OnlineSVR online_svr_tmp;
    OnlineMIMOSVR_ptr mimo_model;

    try {
        mimo_model = online_svr_tmp.load("../SVRRoot/OnlineSVR/test/test_data/5000_mimo_model.txt");
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. Please find what happens " << e.what());
        throw;
        return;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineSVR online_svr_(mimo_model->get_param_set(), mimo_model->get_multistep_len());

    arma::mat learning = mimo_model->get_learning_matrix();
    arma::mat reference = mimo_model->get_reference_matrix();

    LOG4_DEBUG("Learning data rows " << learning.n_rows
                                     << ", reference data rows " << reference.n_rows);

    arma::mat x_train = learning.rows(0, learning.n_rows - 1 - iters);
    arma::mat y_train = reference.rows(0, reference.n_rows - 1 - iters);

    LOG4_DEBUG("Training on data rows " << x_train.n_rows
                                        << ", reference data rows " << y_train.n_rows);

    auto mimo_model_ = online_svr_.get_mimo_model();



    PROFILE_INFO(
            mimo_model_.batch_train(x_train, y_train, true),
            "Batch Train");

    mimo_model_.init_r_matrix();
    arma::mat k = mimo_model_.get_kernel_matrix();
    arma::mat r = mimo_model_.get_r_matrix();

    LOG4_DEBUG("k sizes " << k.n_rows << " " << k.n_cols);
    LOG4_DEBUG("r sizes " << r.n_rows << " " << r.n_cols);

    arma::mat x_learn = learning.rows(learning.n_rows - iters, learning.n_rows - 1);
    arma::mat y_learn = reference.rows(reference.n_rows - iters, reference.n_rows - 1);

    LOG4_DEBUG("Forget and then learn on data rows " << x_learn.n_rows
                                                     << ", reference data rows " << y_learn.n_rows);

    for (size_t i = 0; i < iters; ++i) {
        mimo_model_.forget(0, 0);
        mimo_model_.learn(x_learn.row(i), y_learn.row(i));
    }
    const auto final_k = mimo_model_.get_kernel_matrix();
    const auto final_r = mimo_model_.get_r_matrix();

    const double correction = 1 / mimo_model_.get_param_set().get_svr_C();
//    arma::mat test = final_k * final_r;
//    LOG4_DEBUG("some values are " << test(0, 0) << " " << test(12, 2) << " " << test(44, 11));
    const auto error = error_mult(final_k + arma::eye(k.n_rows,k.n_cols) * correction, final_r);
    LOG4_DEBUG("Error with singular matrix after 100 iterations of forget and learn is " << error);

}
#endif


TEST(mimo_common, fixed_shuffle_returns_the_same){
    arma::uvec i_full = arma::regspace<arma::uvec>(0, 999);
    arma::uvec shuffled1 = svr::common::fixed_shuffle(i_full);
    arma::uvec shuffled2 = svr::common::fixed_shuffle(i_full);
    EXPECT_TRUE(arma::approx_equal(shuffled1, shuffled2, "absdiff", 1));
}

TEST(mimo_common, random_shuffle_returns_different_values){
    arma::mat i_full = arma::cumsum(arma::ones(1000, 1)) - 1;
    const size_t DECON_LEVEL = 4;
    arma::mat shuffled1 = svr::common::shuffle_admat(i_full, DECON_LEVEL);
    arma::mat shuffled2 = svr::common::shuffle_admat(i_full, DECON_LEVEL + 1);
    EXPECT_FALSE(arma::approx_equal(shuffled1, shuffled2, "absdiff", 0.00001));
}

TEST(arma_stuff, t1)
{
    arma::uvec u1 = arma::linspace<arma::uvec>(0, 19, 20);
    size_t ix = 10;
    u1.shed_row(ix);
    u1.transform([ix](double val)
                     { return (val > ix) ? val - 1 : val; });
    arma::uvec six({6});

    u1.print("After shed and move values ul:");
    u1.insert_rows(2, six);
    u1.insert_rows(11, six);
    u1.print("After insert two 6s ul:");
    arma::uvec found_ixs = arma::find(u1 == 6);
    found_ixs.print("print which ixs are equal to 6");
    arma::uvec not_found = arma::find(u1 == 0);

    arma::uvec test({2, 3, 4});
    arma::uvec test2({test.n_rows-1});
//    size_t val = test(test.n_rows);
    arma::uvec found2 = found_ixs;
    arma::uvec found3 = found_ixs;
    found_ixs.insert_rows(found_ixs.n_elem, test.n_rows - 1);
    found_ixs.print("test1");

    found2.insert_rows(found2.n_elem, test2);
    found2.print("test2");

//    found3.insert_rows(found3.n_elem, val);
//    found3.print("test3");

}

#if 0



//HAS SOME MEMORY PROBLEM
void do_mkl_inverse(arma::mat a, arma::mat & inverse_mat)
{
    MKL_INT n = a.n_rows;
    MKL_INT lda = n;
//    std::vector<double> a_(a.n_elem);
//    std::memcpy(a_.data(), a.memptr(), sizeof(double) * a.n_elem);
    std::vector<MKL_INT> ipiv(a.n_rows);
//    MKL_INT info = LAPACKE_dposv( LAPACK_COL_MAJOR, 'U', n, nrhs, a_.data(), lda, b_.data(), ldb );
//    MKL_INT info = LAPACKE_dgesv( LAPACK_COL_MAJOR, n, nrhs, a_.data(), lda, ipiv.data(), b_.data(), ldb );
//    MKL_INT info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a_.data(), lda, ipiv.data());
    MKL_INT info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, const_cast<double *>(a.memptr()), lda, ipiv.data());

//    MKL_INT info = LAPACKE_dpptri('U', n, a_.data());
//    LAPACKE_dpptri('U', n, a_.data(), &info);
    if(info > 0)
    {
        LOG4_DEBUG("The leading minor of order " << info << " is not positive ");
        LOG4_DEBUG("definite; The solution could not be computed.\n");
    }
//    LOG4_DEBUG("copy a_ to x");
//    arma::mat x(a_);
//    x.reshape(a.n_rows, a.n_cols);
//    inverse_mat = x;
//    a_= std::vector<double>();
    inverse_mat =  a;
}

#endif
