#ifndef PGENSEMBLEDAO_HPP
#define PGENSEMBLEDAO_HPP

#include <DAO/EnsembleDAO.hpp>

namespace svr {
namespace dao {

class PgEnsembleDAO : public EnsembleDAO {
public:
    explicit PgEnsembleDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    bigint get_next_id();

    Ensemble_ptr get_by_id(bigint id);

    bool exists(bigint ensemble_id);
    bool exists(const Ensemble_ptr &ensemble);

    Ensemble_ptr get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue);

    std::vector<Ensemble_ptr> find_all_ensembles_by_dataset_id(bigint dataset_id);

    int save(const Ensemble_ptr &Ensemble);

    int remove(const Ensemble_ptr &Ensemble);
};

} }
#endif /* PGENSEMBLEDAO_HPP */

