#include "onlinesvr.hpp"

#include "common/parallelism.hpp"

using namespace svr::common;

namespace svr {

using namespace arma;

size_t OnlineMIMOSVR::learn(const arma::mat &new_x_train, const arma::mat &new_y_train, const bool temp_learn, const bool dont_update_r_matrix, const size_t forget_idx, const bpt::ptime &label_time)
{
#ifdef NO_ONLINE_LEARN
    return new_x_train.n_rows;
#endif

    std::scoped_lock learn_lock(learn_mx);

    if (new_x_train.empty() or new_y_train.empty()) {
        LOG4_WARN("Features are empty " << new_x_train.empty() << " or labels are empty " << new_y_train.empty());
        return 0;
    }
    const auto samples_to_train = new_x_train.n_rows;
    if (samples_to_train > 1) LOG4_THROW("Code not ready for more than one new sample!");

    if (p_svr_parameters->get_kernel_type() == datamodel::kernel_type::DEEP_PATH || p_svr_parameters->get_kernel_type() == datamodel::kernel_type::DEEP_PATH2) {
        if (!p_manifold) LOG4_THROW("Manifold not initialized.");
        arma::mat new_x_train_manifold(new_x_train.n_rows * p_features->n_rows, 1), new_y_train_manifold(new_y_train.n_rows * p_labels->n_rows, 1);
        // Prepare online manifolds.
        THROW_EX_FS(std::logic_error, "Not implemented!");
        for (size_t i = 0; i < new_x_train.n_rows * p_features->n_rows; ++i)
            p_manifold->learn(new_x_train_manifold.row(i), new_y_train_manifold.row(i));
    }

#ifdef PSEUDO_ONLINE
    p_features->insert_rows(p_features->n_rows, new_x_train);
    p_labels->insert_rows(p_labels->n_rows, new_y_train);
    p_features->shed_rows(0, samples_to_train - 1);
    p_labels->shed_rows(0, samples_to_train - 1);
    batch_train(p_features, p_labels, false, {}, true);
    return samples_trained;
#endif

    size_t index_min = std::numeric_limits<size_t>::max();
    {
        arma::mat abs_total_weights = arma::zeros(arma::size(main_components.begin()->second.total_weights));
        for (auto &kv: main_components) abs_total_weights += arma::abs(kv.second.total_weights);
        const auto active_ixs = get_active_ixs();
        if (forget_idx == std::numeric_limits<size_t>::max()) {
            if (temp_idx != std::numeric_limits<size_t>::max()) {
                LOG4_DEBUG("Forgetting temporary row at " << temp_idx);
                index_min = temp_idx;
                temp_idx = std::numeric_limits<size_t>::max();
            }
#ifdef UPDATE_FEATURES // Enable to search for duplicate feature vector
            for (size_t r = 0; r < p_features->n_rows; ++r) {
                if (arma::approx_equal(p_features->row(r), new_x_train, "absdiff", TOL_ROWS)) {
                    index_min = r;
                    LOG4_DEBUG("Found similar features row at " << r);
                }
            }
#endif
            if (index_min == std::numeric_limits<size_t>::max()) {
#ifdef FORGET_MIN_WEIGHT
                const size_t active_index_min = abs_total_weights.rows(active_ixs).index_min();
                index_min = active_ixs(active_index_min, 0);
#else // Forget oldest active index
                index_min = active_ixs.min();
#endif
                LOG4_DEBUG("Forgetting least significant row at " << index_min);
            }
        } else {
            arma::uvec contained_ixs = arma::find(abs_total_weights.rows(active_ixs) == forget_idx);
            if (contained_ixs.empty()) {
                LOG4_WARN("Suggested forget index " << forget_idx << " not found in active indexes, aborting online train.");
                return 0;
            } else {
                LOG4_DEBUG("Using suggested index " << forget_idx);
                index_min = forget_idx;
            }
        }
    }
    p_features->shed_row(index_min);
    p_labels->shed_row(index_min);
    arma::uvec affected_indexes;
    std::vector<size_t> affected_chunks;
    std::mutex mx;
    std::vector<arma::mat> shedded_rows(ixs.size(), arma::mat(main_components.size(), main_components.begin()->second.chunk_weights.front().n_cols));
    __omp_tpfor (size_t, chunk_ix, 0, ixs.size(),
                 auto &chunk_indexes = ixs[chunk_ix];
        const arma::umat find_res = arma::find(chunk_indexes == index_min);
        if (not find_res.is_empty()) {
            const auto chunk_min_ix = *find_res.begin();
            chunk_indexes.shed_row(chunk_min_ix);
            chunk_indexes.rows(arma::find(chunk_indexes >= index_min)) -= 1;

            for (size_t i = 0; main_components.size(); ++i) {
                shedded_rows[chunk_ix].row(i) = main_components[i].chunk_weights[chunk_ix].row(chunk_min_ix);
                main_components[i].chunk_weights[chunk_ix].shed_row(chunk_min_ix);
            }

            if (p_kernel_matrices->at(chunk_ix).n_rows > chunk_min_ix and p_kernel_matrices->at(chunk_ix).n_cols > chunk_min_ix) {
                p_kernel_matrices->at(chunk_ix).shed_row(chunk_min_ix);
                p_kernel_matrices->at(chunk_ix).shed_col(chunk_min_ix);
            } else
                LOG4_ERROR("Failed removing kernel distances for index " << chunk_min_ix << " from kernel matrix for chunk " << chunk_ix << " of size " << arma::size(p_kernel_matrices->at(chunk_ix)));

            std::scoped_lock l(mx);
            affected_indexes.insert_rows(affected_indexes.n_rows, chunk_indexes);
            affected_chunks.push_back(chunk_ix);
            LOG4_DEBUG("index_min " << index_min << " found as chunk_min_ix " << chunk_min_ix << " in ixs[" << chunk_ix << "] " << arma::size(ixs[chunk_ix]));
        } else {
            LOG4_DEBUG("Min index " << index_min << " not found in chunk " << chunk_ix);
            chunk_indexes.rows(arma::find(chunk_indexes > index_min)) -= 1;
        }
    )
    affected_indexes = arma::unique(affected_indexes);
    LOG4_DEBUG("Affected chunks " << affected_chunks.size() << " affected indexes " << arma::size(affected_indexes));
    arma::mat new_kernel_values;
    PROFILE_EXEC_TIME(new_kernel_values = init_predict_kernel_matrix(*p_svr_parameters, p_features->rows(affected_indexes), new_x_train, p_manifold, label_time), "init_predict_kernel_matrix");
    __omp_tpfor (size_t, chunk_ct, 0, affected_chunks.size(),
        const auto chunk_ix = affected_chunks[chunk_ct];
        p_kernel_matrices->at(chunk_ix).resize(p_kernel_matrices->at(chunk_ix).n_rows + 1, p_kernel_matrices->at(chunk_ix).n_cols + 1);
        for (size_t ix_ct = 0; ix_ct < ixs[chunk_ix].size(); ++ix_ct) {
            const auto ix = ixs[chunk_ix](ix_ct, 0);
            const arma::umat f = arma::find(affected_indexes == ix);
            if (f.is_empty()) {
                LOG4_ERROR("Index " << ix << " from chunk " << chunk_ix << " not found in new indexes.");
            } else {
                const auto new_val = new_kernel_values(0, f(0, 0)); // TODO Replace the zero with index of multiple new features vectors
                p_kernel_matrices->at(chunk_ix).at(ix_ct, p_kernel_matrices->at(chunk_ix).n_cols - 1) = new_val;
                p_kernel_matrices->at(chunk_ix).at(p_kernel_matrices->at(chunk_ix).n_rows - 1, ix_ct) = new_val;
            }
        }
        p_kernel_matrices->at(chunk_ix).at(p_kernel_matrices->at(chunk_ix).n_rows - 1, p_kernel_matrices->at(chunk_ix).n_cols - 1) = 1;
        ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, arma::uvec(1, arma::fill::value(p_features->n_rows)));
        //LOG4_DEBUG("ixs[" << chunk_ix << "] " << arma::size(ixs[chunk_ix]));
        for (size_t i = 0; i < main_components.size(); ++i)
            main_components[i].chunk_weights[chunk_ix].insert_rows(
                    main_components[i].chunk_weights[chunk_ix].n_rows,
                    shedded_rows[chunk_ix].row(i));
    )
    p_features->insert_rows(p_features->n_rows, new_x_train);
    p_labels->insert_rows(p_labels->n_rows, new_y_train);
    temp_idx = temp_learn ? p_labels->n_rows - 1 : std::numeric_limits<size_t>::max();

    __omp_pfor_i(0, affected_chunks.size(),
                 const auto chunk_ix = affected_chunks[i];
        const auto labels_chunk = p_labels->rows(ixs[chunk_ix]);

        static size_t auto_gu_ct = 1;
        if (!std::isnormal(auto_gu[chunk_ix]) || !(auto_gu_ct % AUTO_GU_CT))
            PROFILE_EXEC_TIME(go_for_variable_gu(p_kernel_matrices->at(chunk_ix), labels_chunk, auto_gu[chunk_ix]), "go for variable gu on level " << p_svr_parameters->get_decon_level());
        ++auto_gu_ct;

        for (size_t j = 0; j < main_components.size(); ++j) {
            //main_components[j].chunk_weights[chunk_ix].fill(1.);
            calc_weights(chunk_ix, labels_chunk, 1. / (2. * auto_gu[chunk_ix]), main_components[j], IRWLS_ITER_ONLINE);
        }
    )
    update_total_weights();
    samples_trained += samples_to_train;
    return samples_to_train;
}

} // svr
