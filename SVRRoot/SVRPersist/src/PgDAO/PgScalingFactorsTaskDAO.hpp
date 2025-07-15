#ifndef PGSCALINGFACTORSTASKDAO_HPP
#define PGSCALINGFACTORSTASKDAO_HPP

#include "DAO/ScalingFactorsTaskDAO.hpp"

namespace svr {
namespace dao {

class PgScalingFactorsTaskDAO: public ScalingFactorsTaskDAO
{
public:
    explicit PgScalingFactorsTaskDAO(common::PropertiesReader& sql_properties, dao::DataSource& data_source);

    bigint get_next_id();
    bool exists(const bigint id);
    int save(const ScalingFactorsTask_ptr& scalingFactorsTask);
    int remove(const ScalingFactorsTask_ptr& scalingFactorsTask);
    ScalingFactorsTask_ptr get_by_id(const bigint id);
};

}
}

#endif /* PGSCALINGFACTORSTASKDAO_HPP */

