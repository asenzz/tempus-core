#ifndef ASYNCSCALINGFACTORSTASKDAO_H
#define ASYNCSCALINGFACTORSTASKDAO_H

#include <DAO/ScalingFactorsTaskDAO.hpp>

namespace svr {
namespace dao {

class AsyncScalingFactorsTaskDAO : public ScalingFactorsTaskDAO {
    struct AsyncImpl;
    AsyncImpl &pImpl;

public:
    explicit AsyncScalingFactorsTaskDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    ~AsyncScalingFactorsTaskDAO();

    bigint get_next_id();

    bool exists(const bigint id);

    int save(const ScalingFactorsTask_ptr &ScalingFactorsTask);

    int remove(const ScalingFactorsTask_ptr &ScalingFactorsTask);

    ScalingFactorsTask_ptr get_by_id(const bigint id);
};

}
}


#endif /* ASYNCSCALINGFACTORSTASKDAO_H */