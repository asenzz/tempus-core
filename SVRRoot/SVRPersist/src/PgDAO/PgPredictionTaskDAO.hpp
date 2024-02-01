#ifndef PGPREDICTIONTASKDAO_HPP
#define PGPREDICTIONTASKDAO_HPP

#include <DAO/PredictionTaskDAO.hpp>

namespace svr {
namespace dao {

class PgPredictionTaskDAO: public PredictionTaskDAO
{
public:
    explicit PgPredictionTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    bigint get_next_id();
    bool exists(const bigint id);
    int save(const PredictionTask_ptr& predictionTask);
    int remove(const PredictionTask_ptr& predictionTask);
    PredictionTask_ptr get_by_id(const bigint id);
};

}
}

#endif /* PGPREDICTIONTASKDAO_HPP */

