#include <iostream>
#include "util/PerformanceUtils.hpp"
#include "deprecated/OnlineSMOSVR.hpp"


namespace svr {

double OnlineSVR::get_in_sample_training_error() const
{
    auto results = this->predict_inplace_all();
    results.subtract_vector(Y);
    results.abs();
    return results.mean();
}

void OnlineSVR::report_in_sample_training_error() const
{
    auto results = this->predict_inplace_all();
    results.subtract_vector(Y);
    results.abs();
    std::string results_string;
    for (ssize_t i = 0; i < results.size(); ++i) results_string += svr::common::to_string(results.get_value(i)) + " ";
    LOG4_INFO("MAE " << results.mean() << ", absolute inplace residuals " << results_string);
}


bool OnlineSVR::verify_kkt_conditions() const
{

    LOG4_BEGIN();

    const auto smo_epsilon = this->svr_parameters.get_svr_epsilon() / smo_epsilon_divisor;
    std::multimap<double, ssize_t> sorted_violations_weights;
    std::multimap<double, ssize_t> sorted_violations_in_sample_predict;
    bool ris = true;
    // TODO Parallelize
    std::mutex m;
    /*cilk_*/for (ssize_t i = 0; i < Y->size(); ++i) {
        double viol_weight, viol_eps_tube;
        const auto ver_res = verify_kkt_condition(i, viol_weight, viol_eps_tube);
        std::scoped_lock l(m);
        ris &= ver_res;
        sorted_violations_weights.insert(std::make_pair(viol_weight, i));
        sorted_violations_in_sample_predict.insert(std::make_pair(viol_eps_tube, i));
    }
    if (sorted_violations_weights.rbegin()->first > smo_epsilon)
        LOG4_WARN("KKT Weight violation, max: " << sorted_violations_weights.rbegin()->first);
    if (sorted_violations_in_sample_predict.rbegin()->first > smo_epsilon)
        LOG4_WARN("KKT Eps-tube violation, max: " << sorted_violations_in_sample_predict.rbegin()->first);
    const double sum_weights = p_weights->sum();
    if (std::abs(sum_weights) < svr_parameters.get_svr_C() / smo_cost_divisor)
        LOG4_DEBUG("KKT sum of weights " << sum_weights);
    else
        LOG4_WARN("KKT sum of weights violated " << sum_weights << "/"
                                                 << this->svr_parameters.get_svr_C() / smo_cost_divisor);

    ris &= (std::abs(sum_weights) < svr_parameters.get_svr_C() / smo_cost_divisor);

    LOG4_END();

    return ris;
}


bool
OnlineSVR::verify_kkt_condition(
        const ssize_t sample_index,
        double &out_violation_weight,
        double &out_violation_in_sample_predict) const
{
    LOG4_BEGIN();
    const auto svr_c = svr_parameters.get_svr_C();
    const auto svr_epsilon = svr_parameters.get_svr_epsilon();
    const auto H_idx = -margin(sample_index);
    const auto weightsi = p_weights->get_value(sample_index);
    out_violation_weight = std::numeric_limits<double>::max();
    out_violation_in_sample_predict = std::numeric_limits<double>::max();
    const auto smo_epsilon = this->svr_parameters.get_svr_epsilon() / smo_epsilon_divisor;
    const auto smo_weight_epsilon = this->svr_parameters.get_svr_C() / smo_cost_divisor;
    size_t set = 0;
    if (p_support_set_indexes->find(sample_index) >= 0) {
        // Support Set
        set = 1;
        if (weightsi < -svr_c) {
            out_violation_weight = std::abs(weightsi + svr_c);
            out_violation_in_sample_predict = std::max(0., std::abs(H_idx + svr_epsilon));
        } else if (weightsi > svr_c) {
            out_violation_weight = std::abs(weightsi - svr_c);
            out_violation_in_sample_predict = std::max(0., std::abs(svr_epsilon - H_idx));
        } else if (weightsi > smo_weight_epsilon) {
            out_violation_weight = 0.;
            out_violation_in_sample_predict = std::max(0., std::abs(svr_epsilon - H_idx));
        } else if (weightsi < -smo_weight_epsilon) {
            out_violation_weight = 0.;
            out_violation_in_sample_predict = std::max(0., std::abs(H_idx + svr_epsilon));
        } else //if (std::abs(weightsi) < smo_weight_epsilon)
        {
            // How close the point is to any margin?
            out_violation_weight = 0.;
            out_violation_in_sample_predict = std::min(std::abs(svr_epsilon - H_idx), std::abs(H_idx + svr_epsilon));
        }
    } else if (p_error_set_indexes->find(sample_index) >= 0) {
        // Error Set
        set = 2;
        if (weightsi < 0.) {
            out_violation_weight = std::abs(weightsi + svr_c);
            out_violation_in_sample_predict = std::max(0., H_idx + svr_epsilon);
        } else {
            out_violation_weight = std::abs(weightsi - svr_c);
            out_violation_in_sample_predict = std::max(0., svr_epsilon - H_idx);
        }
    } else if (p_remaining_set_indexes->find(sample_index) >= 0) {
        // Remaining Set
        set = 3;
        out_violation_weight = std::abs(weightsi);
        // the margin must be less than eps and more than -eps
        out_violation_in_sample_predict = std::max(0., std::abs(H_idx) - svr_epsilon);
    }

    if (set != 0 && (out_violation_weight > smo_weight_epsilon || out_violation_in_sample_predict > smo_epsilon)) {
        LOG4_WARN(
                "Violated KKT conditions for set " << set << ", sample index " << sample_index << ", weight/cost "
                                                   << weightsi << "/" <<
                                                   svr_c << ", error/eps " << H_idx << "/" << svr_epsilon
                                                   << ", computed violation of weight " <<
                                                   out_violation_weight << ", computed violation of epsilon tube "
                                                   << out_violation_in_sample_predict);
        return false;
    }
    LOG4_END();
    return true;
}


void OnlineSVR::violate_margin(const vektor<double> &h) const
{
    LOG4_BEGIN();
// Check the KKTs margins
#ifdef ONLINE_SVR_DEBUG
    if(this->violate_margin_support(h))
        LOG4_WARN("Violated support set margins!");
    if(this->violate_margin_remaining(h))
        LOG4_WARN("Violated remaining set margins!");
    if(this->violate_margin_error(h))
        LOG4_WARN("Violated error set margins!");
#endif
    LOG4_END();
}

void OnlineSVR::violate_weights() const
{
    LOG4_BEGIN();
// Check the KKTs weights
#ifdef ONLINE_SVR_DEBUG
    if (this->violate_weights_support())
        LOG4_WARN("Violated support set weights!");
    if (this->violate_weights_remaining())
        LOG4_WARN("Violated remaining set weights!");
    if (this->violate_weights_error())
        LOG4_WARN("Violated error set weights!");
#endif
    LOG4_END();
}

bool OnlineSVR::violate_weights_support() const
{
    for (ssize_t ct = 0; ct < p_support_set_indexes->size(); ++ct) {
        const auto ix = (*p_support_set_indexes)[ct];
        if (svr::common::greater(std::abs(p_weights->get_value(ix)), svr_parameters.get_svr_C()))
            return true;
    }
    return false;
}

bool OnlineSVR::violate_weights_remaining() const
{
    for (ssize_t ct = 0; ct < this->p_remaining_set_indexes->size(); ++ct) {
        const auto ix = (*this->p_remaining_set_indexes)[ct];
        if (svr::common::greater(std::abs((*this->p_weights)[ix]), 0.))
            return true;
    }
    return false;
}

bool OnlineSVR::violate_weights_error() const
{
    for (ssize_t ct = 0; ct < this->p_error_set_indexes->size(); ++ct) {
        const auto ix = this->p_error_set_indexes->get_value(ct);
        if (!svr::common::equal_to(std::abs(p_weights->get_value(ix)), this->svr_parameters.get_svr_C()))
            return true;
    }
    return false;
}

bool OnlineSVR::violate_margin_support(const vektor<double> &h) const
{
    for (ssize_t ct = 0; ct < this->p_support_set_indexes->size(); ++ct) {
        const auto ix = this->p_support_set_indexes->get_value(ct);
        if (!svr::common::equal_to(std::abs(h[ix]), this->svr_parameters.get_svr_epsilon(),
                                   this->svr_parameters.get_svr_epsilon() / smo_epsilon_divisor)) {
            LOG4_WARN("violated margin support, index " << ix << ", h: " << h[ix] << ", eps: "
                                                        << this->svr_parameters.get_svr_epsilon()
                                                        << ", tolerance: " << this->svr_parameters.get_svr_epsilon() /
                    smo_epsilon_divisor);
            return true;
        }
    }
    return false;
}

bool OnlineSVR::violate_margin_remaining(const vektor<double> &h) const
{
    for (ssize_t ct = 0; ct < this->p_remaining_set_indexes->size(); ++ct) {
        const auto ix = (*this->p_remaining_set_indexes)[ct];
        if (svr::common::greater(std::abs(h[ix]), this->svr_parameters.get_svr_epsilon(),
                                 this->svr_parameters.get_svr_epsilon() / smo_epsilon_divisor)) {
            LOG4_WARN(
                    "Violated margin for remaining vector " << ix << " weight " << p_weights->get_value(ix)
                                                            << " h " << h[ix] << " eps "
                                                            << svr_parameters.get_svr_epsilon());
            return true;
        }
    }
    return false;
}

bool OnlineSVR::violate_margin_error(const vektor<double> &h) const
{
    for (ssize_t ct = 0; ct < p_error_set_indexes->size(); ++ct) {
        const auto ix = p_error_set_indexes->get_value(ct);
        if (svr::common::less(
                std::abs(h[ix]), svr_parameters.get_svr_epsilon(),
                svr_parameters.get_svr_epsilon() / smo_epsilon_divisor)) {
            LOG4_WARN(
                    "Violated margin error, index " << ix << ", h: " << h[ix] << ", eps: " << svr_parameters.get_svr_epsilon()
                    << ", tolerance: " << svr_parameters.get_svr_epsilon() / smo_epsilon_divisor);
            return true;
        }
    }
    return false;
}

} // namespace svr
