#ifndef ASYNCSCALINGFACTORSTASKDAO_H
#define ASYNCSCALINGFACTORSTASKDAO_H

#include <DAO/ScalingFactorsTaskDAO.hpp>

namespace svr {
namespace dao {

class AsyncScalingFactorsTaskDAO: public ScalingFactorsTaskDAO
{
public:
    explicit AsyncScalingFactorsTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncScalingFactorsTaskDAO();

    bigint get_next_id();
    bool exists(bigint id);
    int save(const ScalingFactorsTask_ptr& ScalingFactorsTask);
    int remove(const ScalingFactorsTask_ptr& ScalingFactorsTask);
    ScalingFactorsTask_ptr get_by_id(bigint id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}



#endif /* ASYNCSCALINGFACTORSTASKDAO_H */

