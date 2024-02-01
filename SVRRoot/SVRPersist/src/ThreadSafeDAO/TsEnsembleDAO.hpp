#pragma once

#include "TsDaoBase.hpp"
#include <DAO/EnsembleDAO.hpp>


namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsEnsembleDAO, EnsembleDAO)

    virtual bigint get_next_id();

    virtual datamodel::Ensemble_ptr get_by_id(const bigint id);

    virtual bool exists(const bigint ensemble_id);
    virtual bool exists(const datamodel::Ensemble_ptr &ensemble);

    virtual datamodel::Ensemble_ptr get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset, const datamodel::DeconQueue_ptr& decon_queue);

    virtual std::deque<datamodel::Ensemble_ptr> find_all_ensembles_by_dataset_id(const bigint dataset_id);

    virtual int save(const datamodel::Ensemble_ptr &Ensemble);

    virtual int remove(const datamodel::Ensemble_ptr &Ensemble);
};

}
}

using EnsembleDAO_ptr = std::shared_ptr <svr::dao::EnsembleDAO>;
