#ifndef PGAUTOTUNEDAO_HPP
#define PGAUTOTUNEDAO_HPP

#include <DAO/AutotuneTaskDAO.hpp>

namespace svr{
namespace dao{

class PgAutotuneTaskDAO: public AutotuneTaskDAO
{
public:
    explicit PgAutotuneTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    bigint get_next_id();

    bool exists(const bigint id);

    int save(const AutotuneTask_ptr& autotune_task);
    int remove(const AutotuneTask_ptr& autotuneTask);

    AutotuneTask_ptr get_by_id(const bigint id);
    std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

} /* namespace dao */
} /* namespace svr */

#endif /* PGAUTOTUNEDAO_HPP */
