#ifndef ASYNCENSEMBEDAO_H
#define ASYNCENSEMBEDAO_H

#include <DAO/EnsembleDAO.hpp>

namespace svr {
namespace dao {

class AsyncEnsembleDAO : public EnsembleDAO
{
public:
    explicit AsyncEnsembleDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncEnsembleDAO();

    virtual bigint get_next_id();

    virtual Ensemble_ptr get_by_id(bigint id);

    virtual bool exists(bigint ensemble_id);
    virtual bool exists(const Ensemble_ptr &ensemble);

    virtual Ensemble_ptr get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue);

    virtual std::vector<Ensemble_ptr> find_all_ensembles_by_dataset_id(bigint dataset_id);

    virtual int save(const Ensemble_ptr &Ensemble);
    virtual int remove(const Ensemble_ptr &Ensemble);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};


} }

#endif /* ASYNCENSEMBEDAO_H */
