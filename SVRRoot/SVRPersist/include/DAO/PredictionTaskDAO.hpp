#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class PredictionTask; } }
using PredictionTask_ptr = std::shared_ptr<svr::datamodel::PredictionTask>;

namespace svr{
namespace dao{

class PredictionTaskDAO: public AbstractDAO
{
public:
    static PredictionTaskDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit PredictionTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(bigint id) = 0;
    virtual int save(const PredictionTask_ptr& predictionTask) = 0;
    virtual int remove(const PredictionTask_ptr& predictionTask) = 0;
    virtual PredictionTask_ptr get_by_id(bigint id) = 0;

};

} /* namespace dao */
} /* namespace svr */

using PredictionTaskDAO_ptr = std::shared_ptr<svr::dao::PredictionTaskDAO>;
