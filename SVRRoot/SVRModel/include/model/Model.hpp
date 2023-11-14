#pragma once

#include <set>
#include <model/Entity.hpp>
#include "relations/ensemble_relation.hpp"

namespace svr {
namespace datamodel {
class SVRParameters;
}
class OnlineMIMOSVR;
}

typedef std::shared_ptr<svr::datamodel::SVRParameters> SVRParameters_ptr;
typedef std::shared_ptr<svr::OnlineMIMOSVR> OnlineMIMOSVR_ptr;

namespace svr {
namespace datamodel {

typedef std::set<size_t, std::less<size_t>> levels_set_t;

class Model : public Entity
{
public:
    bool operator==(Model const &o) const;

    Model() = default;

    Model(bigint id, bigint ensemble_id, size_t decon_level, const std::set<size_t> &learning_levels,
          OnlineMIMOSVR_ptr svr_model, bpt::ptime const &last_modified,
          bpt::ptime const &last_modeled_value_time
    );

    void reset();

    bigint get_ensemble_id() const;

    void set_ensemble_id(const bigint ensemble_id);

    size_t get_decon_level() const;

    void set_decon_level(const size_t decon_level);

    /** Get/set learning wavelet deconstruction level indexes this model is trained against. */
    std::set<size_t> get_learning_levels() const;

    void set_learning_levels(const std::set<size_t> &learning_levels);

    OnlineMIMOSVR_ptr &get_svr_model();

    OnlineMIMOSVR_ptr get_svr_model() const;

    void set_svr_model(OnlineMIMOSVR_ptr svr_model);

    bpt::ptime const &get_last_modified() const;

    void set_last_modified(bpt::ptime const &last_modified);

    bpt::ptime const &get_last_modeled_value_time() const;

    void set_last_modeled_value_time(bpt::ptime const &last_modeled_value_time);

    virtual std::string to_string() const override;

    std::vector<size_t> get_sub_vector_indices() const;

private:
    ensemble_relation ensemble;
    size_t decon_level;
    std::set<size_t> learning_levels;
    OnlineMIMOSVR_ptr svr_model;
    bpt::ptime last_modified {bpt::max_date_time};
    bpt::ptime last_modeled_value_time {bpt::min_date_time};
};

}
}

using Model_ptr = std::shared_ptr<svr::datamodel::Model>;
using Model_vector = std::vector<Model_ptr>;
