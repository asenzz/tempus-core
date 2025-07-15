#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr {
namespace datamodel { class ScalingFactorsTask; }

using ScalingFactorsTask_ptr = std::shared_ptr<datamodel::ScalingFactorsTask>;

namespace dao {

class ScalingFactorsTaskDAO : public AbstractDAO {
public:
    static ScalingFactorsTaskDAO *
    build(common::PropertiesReader &sql_properties, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao);

    explicit ScalingFactorsTaskDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const bigint id) = 0;

    virtual int save(const ScalingFactorsTask_ptr &scalingFactorsTask) = 0;

    virtual int remove(const ScalingFactorsTask_ptr &scalingFactorsTask) = 0;

    virtual ScalingFactorsTask_ptr get_by_id(const bigint id) = 0;

};

using ScalingFactorsTaskDAO_ptr = std::shared_ptr<dao::ScalingFactorsTaskDAO>;

} /* namespace dao */
} /* namespace svr */

