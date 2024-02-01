#ifndef ASYNCPREDICTIONTASKDAO_H
#define ASYNCPREDICTIONTASKDAO_H

#include <DAO/PredictionTaskDAO.hpp>

namespace svr {
namespace dao {

class AsyncPredictionTaskDAO: public PredictionTaskDAO
{
public:
    explicit AsyncPredictionTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncPredictionTaskDAO();

    bigint get_next_id();
    bool exists(const bigint id);
    int save(const PredictionTask_ptr& predictionTask);
    int remove(const PredictionTask_ptr& predictionTask);
    PredictionTask_ptr get_by_id(const bigint id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}



#endif /* ASYNCPREDICTIONTASKDAO_H */

