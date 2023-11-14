#include "deprecated/OnlineSMOSVR.hpp"


using namespace svr::common;
using namespace std;


namespace svr {


// Variation Operations
double
OnlineSVR::find_variation_lc1(
        const vektor<double> &h,
        const vektor<double> &gamma,
        const ssize_t sample_index,
        const int direction) const
{
    // Compute variation new sample --> SupportSet
    const auto weightc = p_weights->get_value(sample_index);
    const auto hc = h[sample_index];
    const auto gammac = gamma[sample_index];
    const auto epsilon = this->svr_parameters.get_svr_epsilon();
    const auto c = this->svr_parameters.get_svr_C();
    const auto smo_eps = svr_parameters.get_svr_epsilon() / smo_epsilon_divisor;
    const auto smo_cost = svr_parameters.get_svr_C() / smo_cost_divisor;

    if (gammac < DEFAULT_T_EPSILON) return direction * INF;

    if (!svr::common::less(hc, +epsilon, smo_eps) && svr::common::less(-c, weightc, smo_cost) && (weightc <= 0.))
        // move to negative margin
        return (-hc + epsilon) / gammac;
    else if (!svr::common::greater(hc, -epsilon, smo_eps) && (0. <= weightc) && svr::common::less(weightc, c, smo_cost))
        return (-hc - epsilon) / gammac;
    else {   // TODO To be handled? Throw an exception!
        LOG4_WARN("Choosing direction failed because the margin does not seem to correspond to the weight. "
                          "sample idx: " << sample_index << ", weightc " <<
                                         weightc << ", hc " << hc << ", gammac " << gammac << ", epsilon " << epsilon
                                         << ", cost " << c);
        return direction * INF;
    }
}


double
OnlineSVR::find_variation_lc2(const ssize_t sample_index, const int direction) const
{
    const auto weightsc = p_weights->get_value(sample_index);
    const auto c = svr_parameters.get_svr_C();
    if (get_support_set_elements_number() == 0)
        return direction * INF;
    else if (direction > 0)
        return -weightsc + c;
    else
        return -weightsc - c;
}


double
OnlineSVR::find_variation_lc(const ssize_t sample_index) const
{
    return -p_weights->get_value(sample_index);
}


void
OnlineSVR::find_variation_ls(
        const vektor<double> &h,
        const vektor<double> &beta,
        const int direction,
        vektor<double> &ls) const
{
    // Variation of support points to error/remaining
    const auto C = svr_parameters.get_svr_C();
    const auto cost_tolerance = C / smo_cost_divisor;
    ls.resize(get_support_set_elements_number());
#ifndef NDEBUG
    {
        vektor<double> supportH;
        vektor<double> support_weights;
        for (auto i = 0; i < this->p_support_set_indexes->size(); ++i) {
            auto support_index = (*this->p_support_set_indexes)[i];
            supportH.add(h[support_index]);
            support_weights.add((*this->p_weights)[support_index]);
        }
        const auto epsilon = this->svr_parameters.get_svr_epsilon();
        const auto epsilon_tolerance = epsilon / smo_epsilon_divisor;
        auto positives_count = supportH.count(epsilon, epsilon_tolerance);
        auto negatives_count = supportH.count(-1. * epsilon, epsilon_tolerance);
        if (positives_count + negatives_count != size_t(supportH.size()))
            LOG4_WARN("Positive epsilons " << positives_count << " and negative epsilons " << negatives_count <<
                                           " does not equal length of margin vector " << supportH.size());
    }
#endif

    /*cilk_*/for (ssize_t el_ix = 0; el_ix < get_support_set_elements_number(); ++el_ix) {
        const auto support_index = this->p_support_set_indexes->get_value(el_ix);
        if (support_index < 0 || support_index >= p_weights->size()) {
            LOG4_ERROR("Invalid index " << support_index << " for element " << el_ix << " in error set!");
        } else {
            const auto support_weight = p_weights->get_value(support_index);
            const auto beta_i = beta[el_ix + 1];
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Support index " << support_index <<  ", support weight " << support_weight << ", beta ix " <<
                       beta_i << " " << el_ix << ", direction " << direction);
#endif
            // const double support_h = h[support_index];
            // TODO: Check if support_h is correct ?!
            double lsi;
            if (svr::common::is_zero(beta_i)) {
                lsi = direction * INF;
#ifdef ONLINE_SVR_DEBUG
                LOG4_DEBUG("Beta equal 0.0, " << beta_i << " supp idx " << el_ix);
                LOG4_DEBUG("case ls00");
#endif //ONLINE_SVR_DEBUG
                LOG4_ERROR("Beta is " << beta_i << ", variation is not defined for point " << support_index);
            } else if (direction * beta_i > 0.) {
                if (0. < support_weight && svr::common::less(support_weight, C, cost_tolerance)) {
                    //Distance to error
                    lsi = (C - support_weight) / beta_i;
                    LOG4_TRACE("case ls001");
                } else if (!svr::common::greater(-C, support_weight) && support_weight < 0.) {
                    // Distance to Remain
                    lsi = -support_weight / beta_i;
                    LOG4_TRACE("case ls002");
                } else {
                    lsi = direction * INF;
#ifdef ONLINE_SVR_DEBUG
                    LOG4_DEBUG("case ls003");
                    LOG4_DEBUG("case ls01");
                    LOG4_WARN("Check KKT 01");
#endif
                    LOG4_WARN("The weight of support index " << support_index << " is not correctly situated for " <<
                                                             " direction * beta." << direction * beta_i << " direction " <<
                                                             direction << ", beta " << beta_i );
                }
            } else if (direction * beta_i < 0.) {
                if (0. < support_weight && !svr::common::greater(support_weight, +C)) {
                    // 0. < support_weight  && support_weight <= +C)
                    LOG4_TRACE("case ls004");
                    lsi = -support_weight / beta_i;
                } else if (!svr::common::greater(-C, support_weight) && support_weight <= 0.) {
                    // -C <= support_weight && support_weight < 0.
                    lsi = (-C - support_weight) / beta_i;
                    LOG4_TRACE("case ls005");
                } else {
                    lsi = direction * INF;
                    LOG4_WARN("The weight of support index " << support_index << " is not correctly situated for direction * beta." <<
                                                             direction * beta_i << " direction " << direction << ", beta " << beta_i);
#ifdef ONLINE_SVR_DEBUG
                    LOG4_DEBUG("case ls006");
                    LOG4_DEBUG("case ls02, supp weight " << support_weight << ", beta " << beta_i << ", direction " << direction );
                    LOG4_WARN("Check KKT 02");
#endif
                }
            } else {
                lsi = direction * INF;
                LOG4_ERROR("Something is weird. This should never happen!");
            }
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Support vector " << support_index << ", weight " << support_weight << ", beta_i " << beta_i << ", variation: " << lsi);
#endif //ONLINE_SVR_DEBUG
            ls[el_ix] = lsi;
        }
    }
#ifdef ONLINE_SVR_DEBUG
    // Check for right sign!
    for (int el_ix = 0; el_ix < ls.get_length(); ++el_ix) {
        if ((int) SIGN(ls[el_ix]) == -SIGN(direction) && !svr::common::equal_to(ls[el_ix], 0.))
            LOG4_ERROR("Something is weird. The value of Ls[" << el_ix << "] = " << ls[el_ix] << " has wrong sign.");
    }
#endif //ONLINE_SVR_DEBUG
}


void
OnlineSVR::find_variation_le(
        const vektor<double> &h,
        const vektor<double> &gamma,
        const int direction,
        vektor<double> &le) const
{
    le.resize(get_error_set_elements_number());
    const auto svr_epsilon = svr_parameters.get_svr_epsilon();
    const auto epsilon_tolerance = svr_epsilon / smo_epsilon_divisor;
    /*cilk_*/for (ssize_t i = 0; i < get_error_set_elements_number(); ++i) {
        const auto error_idx = p_error_set_indexes->get_value(i);
        if (error_idx < 0 || error_idx >= gamma.size()) {
            LOG4_ERROR("Invalid index " << error_idx << " for element " << i << " in error set!");
        } else {
            const auto weightsi = this->p_weights->get_value(error_idx);
            const auto hi = h[error_idx];
            const auto gammai = gamma[this->p_error_set_indexes->get_value(i)];

#ifdef ONLINE_SVR_DEBUG
            double smo_cost_divisor = app_properties_instance.get_smo_cost_divisor();
            const auto svr_c = this->p_svr_parameters.get_svr_C();
            const auto cost_tolerance = svr_c / smo_cost_divisor;
            if (!svr::common::equal_to(ABS(weightsi), +svr_c, cost_tolerance))
                LOG4_ERROR("Wrong error vector weight for error index " << error_idx);

            if (!svr::common::greater(ABS(hi), +svr_epsilon, epsilon_tolerance))
                LOG4_ERROR(
                        "Wrong error vector margin for error index " << error_idx << ". h/eps: " << hi << "/" << svr_epsilon);
#endif //ONLINE_SVR_DEBUG

            double lei;
            if (svr::common::is_zero(direction * gammai, epsilon_tolerance)) {
                LOG4_WARN("Gamma is " << gammai << " for error vector " << error_idx << ", direction is undefined.");
                lei = direction * INF;
            } else if (weightsi > 0. && !svr::common::greater(hi, -svr_epsilon, epsilon_tolerance)) {
                // weight should be +C
                if (direction * gammai > 0.) {
                    // deltaC *currentGamma = delta H > 0
                    lei = (-hi - svr_epsilon) / gammai;
                } else {
                    lei = direction * INF;
                }
            } else if (weightsi < 0. && !svr::common::less(hi, +svr_epsilon)) {
                // weightsi < 0.0 && hi > +svr_epsilon
                // weight should be -C
                if (direction * gammai < 0.) {
                    // deltaC *currentGamma = delta H < 0
                    lei = (-hi + svr_epsilon) / gammai;
                } else {
                    lei = direction * INF;
                }
            } else {
                lei = direction * INF;
                // LOG4_ERROR("The margin is wrong.");
            }
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Error vector " << error_idx << ", weight " << weightsi << ", h " << hi << ", gamma_i " <<
                                       gammai << ", variation " << lei);
#endif //ONLINE_SVR_DEBUG
            le[i] = lei;
        }
    } // main for

    //if sign(val) == -sign(q) and not eq(val,0): #val != 0:
    // TODO: check whether in this condition, we should set Le to Infinity, or it's wrong math or what?
#ifdef ONLINE_SVR_DEBUG
    for (int el_ix = 0; el_ix < le.get_length(); ++el_ix)
        if ((int) SIGN(le[el_ix]) == -SIGN(direction) && !svr::common::equal_to(le[el_ix], 0.))
            LOG4_WARN("Something is weird. The value of Le " << el_ix << " = " << le[el_ix] << " has wrong sign.");
#endif //ONLINE_SVR_DEBUG
}


void
OnlineSVR::find_variation_lr(
        const vektor<double> &h,
        const vektor<double> &gamma,
        const int direction,
        vektor<double> &lr) const
{
    lr.resize(get_remaining_set_elements_number());
    const auto svr_epsilon = svr_parameters.get_svr_epsilon();
    /*cilk_*/for (ssize_t i = 0; i < get_remaining_set_elements_number(); i++) {
        const auto remaining_index = p_remaining_set_indexes->get_value(i);
        if (remaining_index < 0 || remaining_index >= gamma.size()) {
            LOG4_ERROR("Invalid index " << remaining_index << " for element " << i << " in remaining set!");
        } else {
            const auto h_ri = h[remaining_index];
            const auto gamma_ri = gamma[remaining_index];
#ifdef ONLINE_SVR_DEBUG
            if (svr::common::greater(ABS(h_ri), svr_epsilon))
                LOG4_WARN("Something is weird. Element of index " << i << " from remaining set margin/eps: "
                                                                  << h_ri << "/" << svr_epsilon);
#endif // ONLINE_SVR_DEBUG

            double l_ri = direction * INF;
            if (svr::common::equal_to(gamma_ri, 0.)) {
#ifdef ONLINE_SVR_DEBUG
                LOG4_WARN("Gamma is zero, variation is not defined for point " << remaining_index);
#endif //ONLINE_SVR_DEBUG
                l_ri = direction * INF;
            } else if (direction * gamma_ri < 0.) {
                // move the point to the negative margin
                l_ri = (-h_ri - svr_epsilon) / gamma_ri;
            } else if (direction * gamma_ri > 0.) {
                // move the point to the positive margin
                l_ri = (-h_ri + svr_epsilon) / gamma_ri;
            }

#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Remaining vector " << remaining_index << ", weight "
                                           << p_weights->get_value(remaining_index) << ", h " << h_ri
                                           << ", gamma_i " << gamma_ri << ", variation: " << l_ri);
#endif //ONLINE_SVR_DEBUG

            lr[i] = l_ri;
        }
    } //main for
#ifdef ONLINE_SVR_DEBUG
    for (auto el_ix = 0; el_ix < lr.get_length(); ++el_ix)
        if ((int) SIGN(lr[el_ix]) == -SIGN(direction) && !svr::common::equal_to(lr[el_ix], 0.))
            LOG4_WARN("Something is weird. The value of Le[" << el_ix << "] = " << lr[el_ix] << " has wrong sign.");
#endif //ONLINE_SVR_DEBUG
}


void OnlineSVR::find_learning_min_variations(
        const vektor<double> &h,
        const vektor<double> &beta,
        const vektor<double> &gamma,
        const ssize_t sample_index,
        const ssize_t number_variations,
        vektor<double> &min_variations,
        vektor<ssize_t> &min_indexes,
        vektor<ssize_t> &flags) const
{
#ifdef ONLINE_SVR_DEBUG
    const auto weightc = this->p_weights->get_value(sample_index);
    const auto hc = h[sample_index];
#endif //ONLINE_SVR_DEBUG
    const auto gammac = gamma[sample_index];
    auto direction = (int) SIGN(-h[sample_index]);
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Direction " << direction << " -h[sample_index] " << -h[sample_index] << " sample index " << sample_index);
#endif
    // lc1: distance of the new sample to the SupportSet
    const auto lc1 = find_variation_lc1(h, gamma, sample_index, direction);
    direction = (int) SIGN(lc1);
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Direction = " << direction);
    LOG4_DEBUG("Weightc   = " << weightc);
    LOG4_DEBUG("Hc        = " << hc);
    LOG4_DEBUG("Gammac    = " << gammac);
    LOG4_DEBUG("Lc1       = " << lc1);
    if (SIGN(lc1) != SIGN(direction)) LOG4_ERROR("ERROR SIGN!!");
#endif //ONLINE_SVR_DEBUG

    const double direction_inf = direction * INF;
    // distance of the new sample to the ErrorSet
    auto thr_lc2 = std::async(launch::async, &OnlineSVR::find_variation_lc2, this, sample_index, direction);
    vektor<double> ls, le, lr;
    thread thr_ls(&OnlineSVR::find_variation_ls, this, ref(h), ref(beta), direction, ref(ls));
    thread thr_le(&OnlineSVR::find_variation_le, this, ref(h), ref(gamma), direction, ref(le));
    thread thr_lr(&OnlineSVR::find_variation_lr, this, ref(h), ref(gamma), direction, ref(lr));
    thr_ls.join();
    thr_le.join();
    thr_lr.join();
    const double lc2 = thr_lc2.get();

#ifdef ONLINE_SVR_DEBUG
    LOG4_FILE(ONLINESVR_LOG_FILE, "Sample index " << sample_index << " direction  " << direction);

    LOG4_DEBUG("ls:");
    ls.print();
    LOG4_DEBUG("le:");
    le.print();
    LOG4_DEBUG("lr:");
    lr.print();

#endif //ONLINE_SVR_DEBUG

    // Check Values
    if (gammac < 0.) {
        /*cilk_*/for (ssize_t i = 0; i < ls.size(); ++i)
            if (svr::common::is_zero(ls[i])) ls[i] = direction_inf;
        /*cilk_*/for (ssize_t i = 0; i < le.size(); ++i)
            if (svr::common::is_zero(le[i])) le[i] = direction_inf;
        /*cilk_*/for (ssize_t i = 0; i < lr.size(); ++i)
            if (svr::common::is_zero(lr[i])) lr[i] = direction_inf;
    }

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("ls2:");
    ls.print();
    LOG4_DEBUG("le2:");
    le.print();
    LOG4_DEBUG("lr2:");
    lr.print();
#endif //ONLINE_SVR_DEBUG

    for (ssize_t var_idx = 0; var_idx < number_variations; ++var_idx) {
            // Find Min Variation
        double min_ls_value, min_le_value, min_lr_value;
        vektor<double> variations(0., 5);
        ssize_t min_ls_index = 0, min_le_index = 0, min_lr_index = 0;

        if (get_support_set_elements_number() > 0 && ls.size() > 0) {
            ls.min_abs_nonzero(min_ls_value, min_ls_index);
            ls.remove_at(min_ls_index);
        } else
            min_ls_value = direction_inf;

        if (get_error_set_elements_number() > 0 && le.size() > 0) {
            le.min_abs_nonzero(min_le_value, min_le_index);
            le.remove_at(min_le_index);
        } else
            min_le_value = direction_inf;

        if (get_remaining_set_elements_number() > 0 && lr.size() > 0) {
            lr.min_abs_nonzero(min_lr_value, min_lr_index);
            lr.remove_at(min_lr_index);
        } else
            min_lr_value = direction_inf;

        double min_variation = 0.;
        ssize_t min_index = -1;
        ssize_t flag = -1;

        variations[0] = ABS(lc1);
        variations[1] = ABS(lc2);
        variations[2] = min_ls_value;
        variations[3] = min_le_value;
        variations[4] = min_lr_value;
        variations.min_abs(min_variation, flag);

#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Variations: " << variations[0] << ", " << variations[1] << ", " << variations[2] << ", " <<
                   variations[3] << ", " << variations[4] << ", " << "min variation: " << min_variation);
#endif //ONLINE_SVR_DEBUG

        // Find Sample Index Variation
        min_variation *= direction;
        ++flag;
        switch (flag) {
            case 1:
            case 2:
                min_index = 0;
                break;
            case 3:
                min_index = min_ls_index;
                break;
            case 4:
                min_index = min_le_index;
                break;
            case 5:
                min_index = min_lr_index;
                break;
            default:
                min_index = 0;
        }
        min_variations.add(min_variation);
        min_indexes.add(min_index);
        flags.add(flag);
        if (flag != 1 && flag != 2)
        {
            static size_t counts = 0;
            if (svr::common::equal_to(min_variation, 0.)) ++counts;
            else counts = 0;
        }else{
		break;
	}
	
    }
}


void
OnlineSVR::find_learning_min_variation(
        const vektor<double> &h,
        const vektor<double> &beta,
        const vektor<double> &gamma,
        const ssize_t sample_index,
        double &min_variation,
        ssize_t &min_index,
        ssize_t &flag) const
{
    // Find Samples Variations
    int direction;

#ifdef ONLINE_SVR_DEBUG
    const double weightc = this->p_weights->get_value(sample_index);
    const double hc = h[sample_index];
#endif //ONLINE_SVR_DEBUG
    const auto gammac = gamma[sample_index];

    direction = (int) SIGN(-h[sample_index]);
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Direction " << direction << " -h[sample_index] " << -h[sample_index] << " sample index " << sample_index);
#endif
    // lc1: distance of the new sample to the SupportSet
    const double lc1 = find_variation_lc1(h, gamma, sample_index, direction);
    direction = (int) SIGN(lc1);
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Direction = " << direction);
    LOG4_DEBUG("Weightc   = " << weightc);
    LOG4_DEBUG("Hc        = " << hc);
    LOG4_DEBUG("Gammac    = " << gammac);
    LOG4_DEBUG("Lc1       = " << lc1);
    if (SIGN(lc1) != SIGN(direction)) LOG4_ERROR("ERROR SIGN!!");
#endif //ONLINE_SVR_DEBUG

    const double direction_inf = direction * INF;

    // distance of the new sample to the ErrorSet
    auto thr_lc2 = std::async(launch::async, &OnlineSVR::find_variation_lc2, this, sample_index, direction);
    vektor<double> ls, le, lr;
    thread thr_ls(&OnlineSVR::find_variation_ls, this, ref(h), ref(beta), direction, ref(ls));
    thread thr_le(&OnlineSVR::find_variation_le, this, ref(h), ref(gamma), direction, ref(le));
    thread thr_lr(&OnlineSVR::find_variation_lr, this, ref(h), ref(gamma), direction, ref(lr));
    thr_ls.join();
    thr_le.join();
    thr_lr.join();
    const double lc2 = thr_lc2.get();
    if (!lr.size()) LOG4_WARN("Variation LR is zero length.");
    if (!le.size()) LOG4_WARN("Variation LE is zero length.");
    if (!ls.size()) LOG4_WARN("Variation LS is zero length.");

#ifdef ONLINE_SVR_DEBUG
    LOG4_FILE(ONLINESVR_LOG_FILE, "sample index " << sample_index << " Direction  " << direction);

    LOG4_DEBUG("ls:");
    ls.print();
    LOG4_DEBUG("le:");
    le.print();
    LOG4_DEBUG("lr:");
    lr.print();
#endif //ONLINE_SVR_DEBUG

    // Check Values
    if (gammac < 0.) {
        /*cilk_*/for (ssize_t i = 0; i < ls.size(); ++i)
            if (svr::common::is_zero(ls[i])) ls[i] = direction_inf;
        /*cilk_*/for (ssize_t i = 0; i < le.size(); ++i)
            if (svr::common::is_zero(le[i])) le[i] = direction_inf;
        /*cilk_*/for (ssize_t i = 0; i < lr.size(); ++i)
            if (svr::common::is_zero(lr[i])) lr[i] = direction_inf;
    }

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("ls2:");
    ls.print();
    LOG4_DEBUG("le2:");
    le.print();
    LOG4_DEBUG("lr2:");
    lr.print();
#endif //ONLINE_SVR_DEBUG

    double min_ls_value, min_le_value, min_lr_value;
    ssize_t min_ls_index = 0, min_le_index = 0, min_lr_index = 0;

    if (get_support_set_elements_number() > 0 && ls.size() > 0)
        ls.min_abs_nonzero(min_ls_value, min_ls_index);
    else
        min_ls_value = direction_inf;

    if (get_error_set_elements_number() > 0 && le.size() > 0)
        le.min_abs_nonzero(min_le_value, min_le_index);
    else
        min_le_value = direction_inf;

    if (this->get_remaining_set_elements_number() > 0 && lr.size() > 0)
        lr.min_abs_nonzero(min_lr_value, min_lr_index);
    else
        min_lr_value = direction_inf;

    vektor<double> variations(5);
    variations.add(ABS(lc1));
    variations.add(ABS(lc2));
    variations.add(min_ls_value);
    variations.add(min_le_value);
    variations.add(min_lr_value);
    variations.min_abs(min_variation, flag);

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Variations: "
                       << variations[0] << ", "
                       << variations[1] << ", "
                       << variations[2] << ", "
                       << variations[3] << ", "
                       << variations[4] << ", "
                       << "min variation: "
                       << min_variation);
#endif //ONLINE_SVR_DEBUG

    // Find Sample Index Variation
    min_variation *= direction;
    flag++;
    switch (flag) {
        case 1:
        case 2:
            min_index = 0;
            break;
        case 3:
            min_index = min_ls_index;
            break;
        case 4:
            min_index = min_le_index;
            break;
        case 5:
            min_index = min_lr_index;
            break;
        default:
            min_index = 0;
    }
#ifndef NDEBUG
    // PROVE
    static int counts = 0;
    if (svr::common::equal_to(min_variation, 0.)) ++counts;
    else counts = 0;
#endif
}


void OnlineSVR::find_unlearning_min_variations(
        const vektor<double> &h,
        const vektor<double> &beta,
        const vektor<double> &gamma,
        const ssize_t sample_index,
        const ssize_t number_variations,
        vektor<double> &min_variations,
        vektor<ssize_t> &min_indexes,
        vektor<ssize_t> &flags) const
{
    // Find Samples Variations
    // Or, equivalently     int Direction = (int) SIGN( margin );
    const auto direction = (int) SIGN(-p_weights->get_value(sample_index));
    const double direction_inf = direction * INF;

    // Find variation of the sample to the remaining set
    auto thr_lc = std::async(launch::async, &OnlineSVR::find_variation_lc, this, sample_index);
    vektor<double> ls, le, lr;

    thread thr_ls(&OnlineSVR::find_variation_ls, this, ref(h), ref(beta), direction, ref(ls));
    thread thr_le(&OnlineSVR::find_variation_le, this, ref(h), ref(gamma), direction, ref(le));
    thread thr_lr(&OnlineSVR::find_variation_lr, this, ref(h), ref(gamma), direction, ref(lr));
    thr_ls.join();
    thr_le.join();
    thr_lr.join();
    const double lc = thr_lc.get();

    for (ssize_t var_idx = 0; var_idx < number_variations; ++var_idx) {
        vektor<double> variations(0., 4);
        double min_ls_value, min_le_value, min_lr_value;
        ssize_t min_ls_index = 0, min_le_index = 0, min_lr_index = 0;

        if (get_support_set_elements_number() > 0 && ls.size() > 0) {
            ls.min_abs_nonzero(min_ls_value, min_ls_index);
            ls.remove_at(min_ls_index);
            LOG4_DEBUG("Support, min index " << min_ls_index << ", min value " << min_ls_value);
        } else
            min_ls_value = direction_inf;
        // flag 2
        if (get_error_set_elements_number() > 0 && le.size() > 0) {
            le.min_abs_nonzero(min_le_value, min_le_index);
            le.remove_at(min_le_index);
            LOG4_DEBUG("Error, min index " << min_le_index << ", min value " << min_le_value);
        } else
            min_le_value = direction_inf;
        // flag 3
        if (get_remaining_set_elements_number() > 0 && lr.size() > 0) {
            lr.min_abs_nonzero(min_lr_value, min_lr_index);
            lr.remove_at(min_lr_index);
            LOG4_DEBUG("Remain, min index " << min_lr_index << ", min value " << min_lr_value);
        } else
            min_lr_value = direction_inf;

        double min_variation = 0.;
        ssize_t min_index = -1;
        ssize_t flag = -1;

        variations[0] = ABS(lc);
        variations[1] = min_ls_value;
        variations[2] = min_le_value;
        variations[3] = min_lr_value;
        variations.min_abs(min_variation, flag);

        LOG4_DEBUG(
                "Variations " << variations[0] << ", " << variations[1] << ", " << variations[2] << ", " <<
                              variations[3] << ", " << " min variation " << min_variation);

        // Find Sample Index Variation
        min_variation *= direction;
        switch (flag) {
            case 0:
                min_index = 0;
                break;
            case 1:
#ifdef ONLINE_SVR_DEBUG
                LOG4_DEBUG("ls: ");
                for(int i =0; i < ls.get_length(); ++i)
                    LOG4_DEBUG(ls.get_value(i));

                LOG4_DEBUG("min ls value : " << min_ls_value);
                LOG4_DEBUG("min ls index : " << min_ls_index);
#endif //ONLINE_SVR_DEBUG
                min_index = min_ls_index;
                break;

            case 2:
                min_index = min_le_index;
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("le: ");
                for(int i = 0; i < le.get_length(); ++i)
                    LOG4_DEBUG(le.get_value(i));

                LOG4_DEBUG("min le value : " << min_le_value);
                LOG4_DEBUG("min le index : " << min_le_index);
#endif //ONLINE_SVR_DEBUG
                break;


            case 3:
#ifdef ONLINE_SVR_DEBUG
                LOG4_DEBUG("min lr value : " << min_lr_value);
                LOG4_DEBUG("min lr index : " << min_lr_index);
#endif //ONLINE_SVR_DEBUG
                min_index = min_lr_index;
                break;
            default:
                min_index = -1;
        } // switch case
        min_variations.add(min_variation);
        min_indexes.add(min_index);
        flags.add(flag);
#ifdef ONLINE_SVR_DEBUG
        if (svr::common::is_zero(min_variation)) LOG4_WARN("Min variation is 0.");
#endif //ONLINE_SVR_DEBUG
        if (flag == 0) break;
    }
}


void OnlineSVR::find_unlearning_min_variation(
        const vektor<double> &h,
        const vektor<double> &beta,
        const vektor<double> &gamma,
        const ssize_t sample_index,
        double &out_min_variation,
        ssize_t &out_min_index,
        ssize_t &out_flag)
{
    // Find Samples Variations
    // Or, equivalently     int Direction = (int) SIGN( margin );
    const auto direction = (int) SIGN(-this->p_weights->get_value(sample_index));
    const double direction_inf = direction * INF;

    // Find variation of the sample to the remaining set
    auto thr_lc = std::async(launch::async, &OnlineSVR::find_variation_lc, this, sample_index);
    vektor<double> ls, le, lr;
    thread thr_ls(&OnlineSVR::find_variation_ls, this, ref(h), ref(beta), direction, ref(ls));
    thread thr_le(&OnlineSVR::find_variation_le, this, ref(h), ref(gamma), direction, ref(le));
    thread thr_lr(&OnlineSVR::find_variation_lr, this, ref(h), ref(gamma), direction, ref(lr));
    thr_ls.join();
    thr_le.join();
    thr_lr.join();
    const double lc = thr_lc.get();
    if (!lr.size()) LOG4_WARN("Variation LR is zero length.");
    if (!le.size()) LOG4_WARN("Variation LE is zero length.");
    if (!ls.size()) LOG4_WARN("Variation LS is zero length.");

    // Find Min Variation
    double min_ls_value, min_le_value, min_lr_value;
    ssize_t min_ls_index = 0, min_le_index = 0, min_lr_index = 0;
    // flag 1
    if (get_support_set_elements_number() > 0 && ls.size() > 0) {
        ls.min_abs_nonzero(min_ls_value, min_ls_index);
        LOG4_DEBUG("Support, min index " << min_ls_index << ", min value " << min_ls_value);
    } else
        min_ls_value = direction_inf;
    // flag 2
    if (get_error_set_elements_number() > 0 && le.size() > 0) {
        le.min_abs_nonzero(min_le_value, min_le_index);
        LOG4_DEBUG("Error, min index " << min_le_index << ", min value " << min_le_value);
    } else
        min_le_value = direction_inf;
    // flag 3
    if (get_remaining_set_elements_number() > 0 && lr.size() > 0) {
        lr.min_abs_nonzero(min_lr_value, min_lr_index);
        LOG4_DEBUG("Remain, min index " << min_lr_index << ", min value " << min_lr_value);
    } else
        min_lr_value = direction_inf;

    vektor<double> variations(4);
    variations.add(lc);
    variations.add(min_ls_value);
    variations.add(min_le_value);
    variations.add(min_lr_value);
    variations.min_abs(out_min_variation, out_flag);

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("/n lc: " << lc);
    
    LOG4_DEBUG("ls: ");
    for(int i =0; i < ls.get_length(); ++i)
        LOG4_DEBUG(ls.get_value(i));
    LOG4_DEBUG("/n le: ");
    for(int i =0; i < le.get_length(); ++i)
        LOG4_DEBUG(le.get_value(i));
    LOG4_DEBUG("/n lr: ");
    for(int i =0; i < lr.get_length(); ++i)
        LOG4_DEBUG(lr.get_value(i));
#endif //ONLINE_SVR_DEBUG
    // Find Sample Index Variation
    out_min_variation *= direction;
    switch (out_flag) {
        case 0:
            out_min_index = 0;
            break;
        case 1:
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("ls: ");
            for(int i =0; i < ls.get_length(); ++i)
                LOG4_DEBUG(ls.get_value(i));

            LOG4_DEBUG("min ls value : " << min_ls_value);
            LOG4_DEBUG("min ls index : " << min_ls_index);
#endif //ONLINE_SVR_DEBUG
            out_min_index = min_ls_index;
            break;

        case 2:
            out_min_index = min_le_index;
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("le: ");
        for(int i = 0; i < le.get_length(); ++i)
            LOG4_DEBUG(le.get_value(i));

        LOG4_DEBUG("min le value : " << min_le_value);
        LOG4_DEBUG("min le index : " << min_le_index);
#endif //ONLINE_SVR_DEBUG
            break;

        case 3:
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("min lr value : " << min_lr_value);
            LOG4_DEBUG("min lr index : " << min_lr_index);
#endif //ONLINE_SVR_DEBUG
            out_min_index = min_lr_index;
            break;

        default:
            out_min_index = -1;
    }
#ifdef ONLINE_SVR_DEBUG
    if (out_min_index == -1) LOG4_ERROR("Attention, something is wrong!");
    if (svr::common::is_zero(out_min_variation)) LOG4_DEBUG("Attention, MinVariation is 0.");
#endif //ONLINE_SVR_DEBUG
}


// vmatrix R Operations
void
OnlineSVR::update_weights_bias(
        vektor<double> &h,
        vektor<double> &beta,
        vektor<double> &gamma,
        const ssize_t sample_index,
        const double min_variation)
{
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Size of Gamma before updating error + rem H: " << gamma.size()
                                                             << " and of Beta before updating support weights: "
                                                             << beta.size()
                                                             << " out of trained samples: "
                                                             << get_samples_trained_number()
                                                             << " MinVariation " << min_variation);
#endif //ONLINE_SVR_DEBUG
    // Update Weights and Bias
    if (get_support_set_elements_number() > 0) {
        // Update Weights New Sample
        (*p_weights)[sample_index] += min_variation;

        // Update Bias
        const auto delta_weights = beta.clone();
        delta_weights->product_scalar(min_variation);
        this->bias += delta_weights->get_value(0);

        // Update Weights Support Set
        /*cilk_*/for (ssize_t i = 0; i < get_support_set_elements_number(); ++i)
            (*p_weights)[p_support_set_indexes->get_value(i)] += delta_weights->get_value(i + 1);

        delete delta_weights;

        // Update H
        const auto delta_h = gamma.clone();
        delta_h->product_scalar(min_variation);

        /*cilk_*/for (ssize_t i = 0; i < delta_h->size(); ++i)
            // Update margins only for nonsupport points
            if (!p_support_set_indexes->contains(i) && i != sample_index)
                h[i] += delta_h->get_value(i);

        delete delta_h;
    } else {
        // Update Bias
        this->bias += min_variation;

        // Update H
        h.sum_scalar(min_variation);
    }
}

void OnlineSVR::add_sample_to_remaining_set(const ssize_t sample_index)
{
    p_remaining_set_indexes->add(sample_index);
}

// Set Operations
void OnlineSVR::add_to_support_set(
        vektor<double> &h,
        vektor<double> &beta,
        vektor<double> &gamma,
        const ssize_t sample_index,
        const double)
{
    LOG4_DEBUG("Adding sample " << sample_index << " to support set.");
    // Update H and sets
    h[sample_index] = SIGN<double>(h[sample_index]) * this->svr_parameters.get_svr_epsilon();
    p_support_set_indexes->add(sample_index);
    add_to_r_matrix(sample_index, set_type_e::SUPPORT_SET, beta, gamma);
}

void OnlineSVR::add_to_error_set(const int sample_index, double)
{
    LOG4_DEBUG("Add Sample " << sample_index << " to ErrorSet");
    // Update H and Sets
    (*p_weights)[sample_index] = SIGN<double>((*p_weights)[sample_index]) * svr_parameters.get_svr_C();
    this->p_error_set_indexes->add(sample_index);
}

void OnlineSVR::move_from_support_to_error_remaining_set(
        vektor<double> &h,
        vektor<double> &beta,
        vektor<double> &gamma,
        const ssize_t min_index,
        const double min_variation)
{
    const auto index = p_support_set_indexes->get_value(min_index);
    const auto weightsi = p_weights->get_value(index);
    const auto C = svr_parameters.get_svr_C();
    const auto svr_epsilon = svr_parameters.get_svr_epsilon();
    const auto smo_epsilon = svr_epsilon / smo_epsilon_divisor;

#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("Moving point " << index << " from support set to a set with weight " << (*this->p_weights)[index]);
#endif //ONLINE_SVR_DEBUG

    const bool weight_closer_to_zero_than_to_cost = ABS(weightsi) < (C - ABS(weightsi));
    // TODO: Depending on weight_closer_to_zero_than_to_cost,
    // the weight of the point is updated to 0 or +/-C.
    // This breaks the KKT weights sum with no stabilization.
    if (weight_closer_to_zero_than_to_cost) {
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Moving to remaining");
        LOG4_DEBUG("Case 3a");
#endif //ONLINE_SVR_DEBUG
        (*p_weights)[index] = 0.;
        // CASE 3a: Move Sample from SupportSet to RemainingSet
        LOG4_DEBUG("Move sample " << index << " from support set to remaining set.");
        // Update H and Sets
        p_remaining_set_indexes->add(index);
        p_support_set_indexes->remove_at(min_index);
        remove_from_r_matrix(min_index);

        // This is an artificial way to fix the margin for the next sample learning -->
        // Not handled in the original implementation and the (official) paper.
        const auto new_margin_predicted = this->margin(this->X->get_row_ref(index), this->Y->get_value(index));
        const auto new_margin_from_update_rule = h[index] + gamma[index] * min_variation;
        if (ABS(new_margin_predicted) < svr_epsilon - smo_epsilon) {
            h[index] = new_margin_predicted;
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3a1 h [" << index << "] =  " << new_margin_predicted);
            LOG4_DEBUG("epsilon " << svr_epsilon << ", smo_epsilon " << smo_epsilon);
            LOG4_DEBUG("Case 3a1");
#endif //ONLINE_SVR_DEBUG
        } else if (ABS(new_margin_from_update_rule) < svr_epsilon - smo_epsilon) {
            h[index] = new_margin_from_update_rule;
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3a3 h [" << index << "] =  " << new_margin_from_update_rule);
            LOG4_DEBUG("Case 3a2");
#endif //ONLINE_SVR_DEBUG
        } else {
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3a3");
#endif //ONLINE_SVR_DEBUG
            // Try with the updated values of support / remaining sets
            const auto beta_ = find_beta(index);
            const auto gamma_ = find_gamma(beta_, index);
            const auto margin_gamma_ = h[index] + gamma_[index] * min_variation;
            if (ABS(margin_gamma_) < svr_epsilon - smo_epsilon) {
#ifdef ONLINE_SVR_DEBUG
                LOG4_DEBUG("Case 3a31");
                LOG4_DEBUG("Case 3a4 h [" << index << "] =  " << margin_gamma_);
#endif //ONLINE_SVR_DEBUG
                h[index] = margin_gamma_;
            }
        }

        // Very last option to keep the margin correct for the next iteration!
        if (ABS(h[index]) >= svr_epsilon - smo_epsilon) {
            h[index] = -(int) SIGN(weightsi) * svr_epsilon / 2.; // TODO Evgeniy Check
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3a5 h [" << index << "] =  " << h[index]);
            LOG4_DEBUG("Case 3a951");
#endif //ONLINE_SVR_DEBUG
        }
    } else {
#ifdef ONLINE_SVR_DEBUG
        LOG4_DEBUG("Moving to error");
        LOG4_DEBUG("Case 3b");
        LOG4_DEBUG("Move Sample " << index << " from support set to error set.");
#endif //ONLINE_SVR_DEBUG
        (*p_weights)[index] = SIGN(weightsi) * C;
        // CASE 3b: Move Sample from SupportSet to ErrorSet
        // Update H and Sets
        p_error_set_indexes->add(index);
        p_support_set_indexes->remove_at(min_index);
        remove_from_r_matrix(min_index);

        const auto new_margin = this->margin(X->get_row_ref(index), Y->get_value(index));
        const auto margin_gamma = h[index] + gamma[index] * min_variation;
        // abs(margin) > self.eps and sign(margin) != sign(weightsValue)
        if ((ABS(margin_gamma) > svr_epsilon + smo_epsilon) && (int) SIGN(margin_gamma) != (int) SIGN(weightsi)) {
#ifdef  ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3b2");
#endif //ONLINE_SVR_DEBUG
            h[index] = margin_gamma;
        } else if ((ABS(new_margin) > svr_epsilon + smo_epsilon) && (int) SIGN(new_margin) != (int) SIGN(weightsi)) {
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3b1");
#endif //ONLINE_SVR_DEBUG
            h[index] = new_margin;
        } else {
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3b3");
#endif //ONLINE_SVR_DEBUG
            // Try with the updated values of support / remaining sets
            const auto beta_ = find_beta(index);
            const auto gamma_ = find_gamma(beta_, index);
            const auto margin_gamma_ = h[index] + gamma_[index] * min_variation;
            if ((ABS(margin_gamma_) > svr_epsilon + smo_epsilon) && (int) SIGN(margin_gamma_) != (int) SIGN(weightsi)) {
#ifdef ONLINE_SVR_DEBUG
                LOG4_DEBUG("Case 3b31");
#endif //ONLINE_SVR_DEBUG
                h[index] = margin_gamma_;
            }
        }

        // Very last option to keep the margin correct for the next iteration!
        if (ABS(h[index]) <= svr_epsilon + smo_epsilon) {
#ifdef ONLINE_SVR_DEBUG
            LOG4_DEBUG("Case 3b951");
#endif //ONLINE_SVR_DEBUG
            h[index] = (-SIGN(weightsi) * svr_epsilon * 3.) / 2.;
        }
    }
}

void
OnlineSVR::move_from_error_to_support_set(
        vektor<double> &h,
        vektor<double> &beta,
        vektor<double> &gamma,
        const ssize_t min_index,
        const double)
{

    const auto index = p_error_set_indexes->get_value(min_index);
    // Update H and Sets
    LOG4_DEBUG("Move sample " << index << " from error set to support set.");
    h[index] = SIGN<double>(h[index]) * this->svr_parameters.get_svr_epsilon();
    p_support_set_indexes->add(index);
    p_error_set_indexes->remove_at(min_index);
    add_to_r_matrix(index, set_type_e::ERROR_SET, beta, gamma);
}


void
OnlineSVR::move_from_remaining_to_support_set(
        vektor<double> &h,
        vektor<double> &beta,
        vektor<double> &gamma,
        const ssize_t min_index,
        const double)
{
    const auto index = p_remaining_set_indexes->get_value(min_index);
    // Update H and Sets
    LOG4_DEBUG("Move Sample " << index << " from remaining set to support set.");
    h[index] = SIGN<double>(h[index]) * this->svr_parameters.get_svr_epsilon();
    this->p_support_set_indexes->add(index);
    this->p_remaining_set_indexes->remove_at(min_index);
    this->add_to_r_matrix(index, set_type_e::REMAINING_SET, beta, gamma);
}


void OnlineSVR::remove_from_support_set(const ssize_t sample_set_index)
{
    p_support_set_indexes->remove_at(sample_set_index);
    remove_from_r_matrix(sample_set_index);
}


void OnlineSVR::remove_from_error_set(const ssize_t sample_set_index)
{
    p_error_set_indexes->remove_at(sample_set_index);
}

void OnlineSVR::remove_from_remaining_set(const ssize_t sample_set_index)
{
    p_remaining_set_indexes->remove_at(sample_set_index);
}


void OnlineSVR::reorder_set_indexes_decreasing(const ssize_t sample_index)
{
    /* cilk_ */ for (ssize_t i = 0; i < this->get_support_set_elements_number(); ++i) {
        auto &ssi = (*this->p_support_set_indexes)[i];
        if (ssi > sample_index) ssi -= 1;
    }
    /* cilk_ */ for (ssize_t i = 0; i < this->get_error_set_elements_number(); ++i) {
        auto &esi = (*this->p_error_set_indexes)[i];
        if (esi > sample_index) esi -= 1;
    }
    /* cilk_ */ for (ssize_t i = 0; i < this->get_remaining_set_elements_number(); ++i) {
        auto &rsi = (*this->p_remaining_set_indexes)[i];
        if (rsi > sample_index) rsi -= 1;
    }
}

void OnlineSVR::remove_sample(const ssize_t sample_index)
{
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG(
        "Remove sample_index " << sample_index << " insample margin size " << in_sample_margins.size() <<
        " number samples trained " << this->samples_trained_number);
#endif /* #ifdef ONLINE_SVR_DEBUG */
    X->remove_row(sample_index);
    Y->remove_at(sample_index);
    p_weights->remove_at(sample_index);
    if (save_kernel_matrix) remove_from_kernel_matrix(sample_index);

    reorder_set_indexes_decreasing(sample_index);
    in_sample_margins.remove_at(sample_index);
    --samples_trained_number;
    if (samples_trained_number == 1 && get_error_set_elements_number() > 0) {
        p_error_set_indexes->remove_at(0);
        p_remaining_set_indexes->add(0);
        (*p_weights)[0] = 0.;
        bias = static_cast<decltype(bias)>(margin(X->get_row_ref(0), Y->get_value(0)));
    }
    if (samples_trained_number < 1) bias = 0.;
#ifdef ONLINE_SVR_DEBUG
    LOG4_DEBUG("After remove insample margin lenght is " << in_sample_margins.size() <<
               " number samples trained is " << samples_trained_number);
#endif /* #ifdef ONLINE_SVR_DEBUG */
}

} // svr
