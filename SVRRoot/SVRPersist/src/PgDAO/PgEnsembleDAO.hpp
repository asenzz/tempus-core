#ifndef PGENSEMBLEDAO_HPP
#define PGENSEMBLEDAO_HPP

#include "DAO/EnsembleDAO.hpp"

namespace svr {
namespace dao {

class PgEnsembleDAO : public EnsembleDAO {
public:
    explicit PgEnsembleDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    bigint get_next_id();

    datamodel::Ensemble_ptr get_by_id(const bigint id);

    bool exists(const bigint ensemble_id);

    bool exists(const datamodel::Ensemble_ptr &ensemble);

    datamodel::Ensemble_ptr get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset, const datamodel::DeconQueue_ptr &decon_queue);

    std::deque<datamodel::Ensemble_ptr> find_all_ensembles_by_dataset_id(const bigint dataset_id);

    int save(const datamodel::Ensemble_ptr &Ensemble);

    int remove(const datamodel::Ensemble_ptr &Ensemble);
};

}
}
#endif /* PGENSEMBLEDAO_HPP */

