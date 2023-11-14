#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel { class IQScalingFactor; } }
using IQScalingFactor_ptr = std::shared_ptr<svr::datamodel::IQScalingFactor>;

namespace svr {
namespace dao {

class IQScalingFactorDAO : public AbstractDAO
{
public:
    static IQScalingFactorDAO* build(svr::common::PropertiesFileReader& sqlProperties,
                                     svr::dao::DataSource& dataSource,
                                     svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit IQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties,
                                svr::dao::DataSource& dataSource);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const IQScalingFactor_ptr& iQscalingFactor) = 0;
    virtual int remove(const IQScalingFactor_ptr& iQscalingFactor) = 0;
    virtual std::vector<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id) = 0;
};

}
}

