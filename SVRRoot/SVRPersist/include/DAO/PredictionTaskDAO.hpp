#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr {

namespace datamodel { class PredictionTask; }

using PredictionTask_ptr = std::shared_ptr<datamodel::PredictionTask>;

namespace dao {

class PredictionTaskDAO: public AbstractDAO
{
public:
    static PredictionTaskDAO *build(common::PropertiesReader &sql_properties, dao::DataSource& data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao);

    explicit PredictionTaskDAO(common::PropertiesReader& sql_properties, dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const PredictionTask_ptr& predictionTask) = 0;
    virtual int remove(const PredictionTask_ptr& predictionTask) = 0;
    virtual PredictionTask_ptr get_by_id(const bigint id) = 0;
};

using PredictionTaskDAO_ptr = std::shared_ptr<dao::PredictionTaskDAO>;

} /* namespace dao */
} /* namespace svr */