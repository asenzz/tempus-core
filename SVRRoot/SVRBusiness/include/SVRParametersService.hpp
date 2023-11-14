#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class  SVRParametersDAO; } }

namespace svr { namespace datamodel { class  SVRParameters; } }
using SVRParameters_ptr = std::shared_ptr<svr::datamodel::SVRParameters>;

namespace svr {
namespace business {

class SVRParametersService {

    svr::dao::SVRParametersDAO & svr_parameters_dao;

public:
    SVRParametersService(svr::dao::SVRParametersDAO &svr_parameters_dao):
        svr_parameters_dao(svr_parameters_dao)
    {}

    bool exists(const SVRParameters_ptr &svr_parameters);
    bool exists(const bigint svr_parameters_id);
    int save(const svr::datamodel::SVRParameters &svr_parameters);
    int save(const SVRParameters_ptr &svr_parameters);
    int remove(const SVRParameters_ptr &svr_parameters);
    int remove(const svr::datamodel::SVRParameters &svr_parameters);
    int remove(const bigint svr_parameters_id);
    int remove_by_dataset(const bigint dataset_id);

    std::map<std::pair<std::string, std::string>, std::vector<SVRParameters_ptr> > get_all_by_dataset_id(const bigint dataset_id);
private:
};

}
}

using SVRParametersService_ptr = std::shared_ptr<svr::business::SVRParametersService>;
