#ifndef ASYNCAUTOTUNEDAO_HPP
#define ASYNCAUTOTUNEDAO_HPP

#include <DAO/AutotuneTaskDAO.hpp>

namespace svr { namespace dao {

class AsyncAutotuneTaskDAO: public AutotuneTaskDAO
{
public:
    explicit AsyncAutotuneTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncAutotuneTaskDAO();

    bigint get_next_id();

    bool exists(bigint id);

    int save(const AutotuneTask_ptr& autotuneTask);
    int remove(const AutotuneTask_ptr& autotuneTask);

    AutotuneTask_ptr get_by_id(bigint id);
    std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}


#endif /* ASYNCAUTOTUNEDAO_HPP */

