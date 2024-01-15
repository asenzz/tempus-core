#include <iostream>
#include "onlinesvr.hpp"

#include <tbb/parallel_for.h>


namespace svr {


void OnlineMIMOSVR::best_forget(const bool dont_update_r_matrix)
{
    LOG4_THROW("Not used any longer, learn does best forget.");

    size_t forget_from = 0;
    size_t max_size = ixs[0].n_elem;
    for (size_t i = 1; i < ixs.size(); ++i) {
        if (ixs[i].n_elem > max_size) {
            forget_from = i;
            max_size = ixs[i].n_elem;
        }
    }
    arma::vec weight_sums;
    for (auto &kv: main_components) {
        auto &component = kv.second;
        arma::mat first_weights = component.chunk_weights[forget_from].rows(
                0, component.chunk_weights[forget_from].n_rows / 2);
        weight_sums.resize(first_weights.n_rows);
        weight_sums += arma::sum(arma::pow(first_weights, 2), 1);
    }
    size_t min_index = weight_sums.index_min();
    size_t real_ix = ixs[forget_from](min_index);

    std::vector<arma::uvec> found_ixs(ixs.size());
    /*cilk_*/for (size_t i = 0; i < ixs.size(); ++i) {
        found_ixs[i] = arma::find(ixs[i] == real_ix);
        for (size_t j = 0; j < found_ixs[i].n_elem; ++j)
            forget(found_ixs[i](j), i, dont_update_r_matrix);
    }

    // Delete from learning and reference matrix and update new indexes after that
    p_features->shed_row(real_ix);
    p_labels->shed_row(real_ix);

    update_ixs(real_ix);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, ixs.size()), [&] (const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            //if (auto_gu[i] == 0.) THROW_EX_FS(std::logic_error, "Auto GU not initialized for index group " << i);
            const double correction = 1. / (2. * this->auto_gu[i]);
            if (found_ixs[i].n_elem > 0) {
                for (auto &kv: main_components) {
                    if (!dont_update_r_matrix)
                        kv.second.chunk_weights[i] = r_matrix[i] * (p_labels->rows(ixs[i]) + kv.second.epsilon);
                    else {
                        arma::mat singular = arma::eye(p_kernel_matrices->at(i).n_rows,
                                                       p_kernel_matrices->at(i).n_cols);
                        arma::mat kernel_chunk = p_kernel_matrices->at(i) + singular * correction;
                        arma::mat reference_chunk = p_labels->rows(ixs[i]) + kv.second.epsilon;
                        /*
                        PROFILE_EXEC_TIME(
                                kv.second.chunk_weights[i] = do_gpu_solve(
                                        kernel_chunk.mem, reference_chunk.mem,
                                        kernel_chunk.n_rows, reference_chunk.n_cols),
                                "do_ocl_solve online forget");
                        */
                        /*
                        solve_dispatch(
                                correction,
                                p_kernel_matrices->at(i) + singular * correction,
                                p_labels->rows(ixs[i]) + kv.second.epsilon,
                                kv.second.chunk_weights[i]);
                        */
                    }
                }
            }
        }
    });

    update_total_weights();
    --samples_trained;
}

void OnlineMIMOSVR::forget(const size_t inner_ix, const size_t outer_ix, const bool dont_update_r_matrix)
{
    arma::mat old_kernel_matrix = p_kernel_matrices->at(outer_ix);
    if (not dont_update_r_matrix) PROFILE_EXEC_TIME(
            remove_from_r_matrix(inner_ix, outer_ix), "Remove from R matrix.");
    p_kernel_matrices->at(outer_ix).shed_row(inner_ix);
    p_kernel_matrices->at(outer_ix).shed_col(inner_ix);
    ixs[outer_ix].shed_row(inner_ix);
}

void OnlineMIMOSVR::update_ixs(const size_t ix)
{
    for (auto &ix_vec: ixs)
        ix_vec.transform([ix](double val)
                         { return (val > ix) ? val - 1 : val; });
}

} // svr

