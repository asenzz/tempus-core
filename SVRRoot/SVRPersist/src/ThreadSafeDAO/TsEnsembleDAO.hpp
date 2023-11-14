#pragma once

#include "TsDaoBase.hpp"
#include <DAO/EnsembleDAO.hpp>


namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsEnsembleDAO, EnsembleDAO)

    virtual bigint get_next_id();

    virtual Ensemble_ptr get_by_id(bigint id);

    virtual bool exists(bigint ensemble_id);
    virtual bool exists(const Ensemble_ptr &ensemble);

    virtual Ensemble_ptr get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue);

    virtual std::vector<Ensemble_ptr> find_all_ensembles_by_dataset_id(bigint dataset_id);

    virtual int save(const Ensemble_ptr &Ensemble);

    virtual int remove(const Ensemble_ptr &Ensemble);
};

}
}

using EnsembleDAO_ptr = std::shared_ptr <svr::dao::EnsembleDAO>;
