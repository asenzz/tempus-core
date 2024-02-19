#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class AutotuneTask; } }
using AutotuneTask_ptr = std::shared_ptr<svr::datamodel::AutotuneTask>;

namespace svr{
namespace dao{

class AutotuneTaskDAO: public AbstractDAO
{
public:
    static AutotuneTaskDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit AutotuneTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const AutotuneTask_ptr& autotuneTask) = 0;
    virtual int remove(const AutotuneTask_ptr& autotuneTask) = 0;
    virtual AutotuneTask_ptr get_by_id(const bigint id) = 0;
    virtual std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id) = 0;
};

} /* namespace dao */
} /* namespace svr */

using AutotuneTaskDAO_ptr = std::shared_ptr<svr::dao::AutotuneTaskDAO>;
