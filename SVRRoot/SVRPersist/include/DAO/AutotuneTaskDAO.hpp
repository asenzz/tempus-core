#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class AutotuneTask; }}

namespace svr {
namespace dao {

using AutotuneTask_ptr = std::shared_ptr<datamodel::AutotuneTask>;

class AutotuneTaskDAO : public AbstractDAO {
public:
    static AutotuneTaskDAO *build(common::PropertiesReader &sql_properties, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao);

    explicit AutotuneTaskDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const bigint id) = 0;

    virtual int save(const AutotuneTask_ptr &autotuneTask) = 0;

    virtual int remove(const AutotuneTask_ptr &autotuneTask) = 0;

    virtual AutotuneTask_ptr get_by_id(const bigint id) = 0;

    virtual std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id) = 0;
};

using AutotuneTaskDAO_ptr = std::shared_ptr<dao::AutotuneTaskDAO>;

} /* namespace dao */
} /* namespace svr */
