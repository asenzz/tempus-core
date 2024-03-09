//
// Created by zarko on 1/13/24.
//

#include "model/SVRParameters.hpp"

namespace svr {
namespace datamodel {

kernel_type_e operator++(kernel_type_e &k_type)
{
    int tmp = static_cast<int>(k_type);
    tmp++;
    return k_type = static_cast<kernel_type_e >( tmp);
}

kernel_type_e operator++(kernel_type_e &k_type, int)
{
    kernel_type_e tmp(k_type);
    ++k_type;
    return tmp;
}

bool less_SVRParameters_ptr::operator()(const datamodel::SVRParameters_ptr &lhs, const datamodel::SVRParameters_ptr &rhs) const
{
    return lhs->operator < (*rhs);
}


SVRParameters::SVRParameters(
        const bigint id,
        const bigint dataset_id,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name,
        const size_t decon_level,
        const size_t chunk_ix,
        const size_t grad_level,
        const double svr_C,
        const double svr_epsilon,
        const double svr_kernel_param,
        const double svr_kernel_param2,
        const u_int64_t svr_decremental_distance,
        const double svr_adjacent_levels_ratio,
        const kernel_type_e kernel_type,
        const size_t lag_count)
        : Entity(id),
          dataset_id(dataset_id),
          input_queue_table_name(input_queue_table_name),
          input_queue_column_name(input_queue_column_name),
          decon_level_(decon_level),
          chunk_ix_(chunk_ix),
          grad_level_(grad_level),
          svr_C(svr_C),
          svr_epsilon(svr_epsilon),
          svr_kernel_param(svr_kernel_param),
          svr_kernel_param2(svr_kernel_param2),
          svr_decremental_distance(svr_decremental_distance),
          svr_adjacent_levels_ratio(svr_adjacent_levels_ratio),
          kernel_type(kernel_type),
          lag_count(lag_count)
{
    set_kernel_type(kernel_type);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

SVRParameters::SVRParameters(const SVRParameters &params) :
        SVRParameters(
                bigint(0),
                params.get_dataset_id(),
                params.get_input_queue_table_name(),
                params.get_input_queue_column_name(),
                params.get_decon_level(),
                params.get_chunk_ix(),
                params.get_grad_level(),
                params.get_svr_C(),
                params.get_svr_epsilon(),
                params.get_svr_kernel_param(),
                params.get_svr_kernel_param2(),
                params.get_svr_decremental_distance(),
                params.get_svr_adjacent_levels_ratio(),
                params.get_kernel_type(),
                params.get_lag_count())
{
    set_kernel_type(kernel_type);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

SVRParameters &SVRParameters::operator=(const SVRParameters &v)
{
    set_id(v.get_id());
    dataset_id = v.dataset_id;
    input_queue_table_name = v.input_queue_table_name;
    input_queue_column_name = v.input_queue_column_name;
    decon_level_ = v.decon_level_;
    chunk_ix_ = v.chunk_ix_;
    grad_level_ = v.grad_level_;
    svr_C = v.svr_C;
    svr_epsilon = v.svr_epsilon;
    svr_kernel_param = v.svr_kernel_param;
    svr_kernel_param2 = v.svr_kernel_param2;
    svr_decremental_distance = v.svr_decremental_distance;
    svr_adjacent_levels_ratio = v.svr_adjacent_levels_ratio;
    kernel_type = v.kernel_type;
    lag_count = v.lag_count;
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    return *this;
}

void SVRParameters::init_id()
{
    if (!id) {
        if (input_queue_table_name.size()) boost::hash_combine(id, input_queue_table_name);
        if (input_queue_column_name.size()) boost::hash_combine(id, input_queue_column_name);
        boost::hash_combine(id, decon_level_);
        boost::hash_combine(id, grad_level_);
        boost::hash_combine(id, chunk_ix_);
    }
}

bool SVRParameters::operator==(const SVRParameters &o) const
{
    return dataset_id == o.dataset_id
           && input_queue_table_name == o.input_queue_table_name
           && input_queue_column_name == o.input_queue_column_name
           && decon_level_ == o.decon_level_
           && chunk_ix_ == o.chunk_ix_
           && grad_level_ == o.grad_level_
           && svr_C == o.svr_C
           && svr_epsilon == o.svr_epsilon
           && svr_kernel_param == o.svr_kernel_param
           && svr_kernel_param2 == o.svr_kernel_param2
           && svr_decremental_distance == o.svr_decremental_distance
           && svr_adjacent_levels_ratio == o.svr_adjacent_levels_ratio
           && kernel_type == o.kernel_type
           && lag_count == o.lag_count;
}

bool SVRParameters::operator!=(const SVRParameters &o) const
{
    return !operator==(o);
}

bool SVRParameters::operator<(const SVRParameters &o) const
{
    if (input_queue_table_name < o.input_queue_table_name) return true;
    if (input_queue_table_name > o.input_queue_table_name) return false;

    if (input_queue_column_name < o.input_queue_column_name) return true;
    if (input_queue_column_name > o.input_queue_column_name) return false;

    if (decon_level_ < o.decon_level_) return true;
    if (decon_level_ > o.decon_level_) return false;

    if (grad_level_ < o.grad_level_) return true;
    if (grad_level_ > o.grad_level_) return false;

    return chunk_ix_ < o.chunk_ix_;
}

bigint SVRParameters::get_dataset_id() const
{
    return dataset_id;
}

void SVRParameters::set_dataset_id(const bigint &value)
{
    dataset_id = value;
}

std::string SVRParameters::get_input_queue_column_name() const
{
    return input_queue_column_name;
}

void SVRParameters::set_input_queue_column_name(const std::string &value)
{
    input_queue_column_name = value;
}

std::string SVRParameters::get_input_queue_table_name() const
{
    return input_queue_table_name;
}

void SVRParameters::set_input_queue_table_name(const std::string &value)
{
    input_queue_table_name = value;
}

size_t SVRParameters::get_decon_level() const
{
    return decon_level_;
}

void SVRParameters::set_decon_level(const size_t _decon_level)
{
    decon_level_ = _decon_level;
}

size_t SVRParameters::get_chunk_ix() const
{
    return chunk_ix_;
}

void SVRParameters::set_chunk_ix(const size_t _chunk_ix)
{
    chunk_ix_ = _chunk_ix;
}

size_t SVRParameters::get_grad_level() const
{
    return grad_level_;
}

void SVRParameters::set_grad_level(const size_t _grad_level)
{
    grad_level_ = _grad_level;
}

void SVRParameters::decrement_gradient()
{
    if (grad_level_) --grad_level_;
}

double SVRParameters::get_svr_C() const
{
    return svr_C;
}

void SVRParameters::set_svr_C(const double _svr_C)
{
    svr_C = _svr_C;
}

double SVRParameters::get_svr_epsilon() const
{
    return svr_epsilon;
}


void SVRParameters::set_svr_epsilon(const double _svr_epsilon)
{
    if (_svr_epsilon < 0) THROW_EX_FS(std::invalid_argument, "Epsilon parameter " << _svr_epsilon << " is less than zero.");
    svr_epsilon = _svr_epsilon;
}


double SVRParameters::get_svr_kernel_param() const
{
    return svr_kernel_param;
}

void SVRParameters::set_svr_kernel_param(const double _svr_kernel_param)
{
    svr_kernel_param = _svr_kernel_param;
}

double SVRParameters::get_svr_kernel_param2() const
{
    return svr_kernel_param2;
}

void SVRParameters::set_svr_kernel_param2(const double _svr_kernel_param2)
{
    svr_kernel_param2 = _svr_kernel_param2;
}

u_int64_t SVRParameters::get_svr_decremental_distance() const
{
    return svr_decremental_distance;
}


void SVRParameters::set_svr_decremental_distance(const uint64_t _svr_decremental_distance)
{
    svr_decremental_distance = _svr_decremental_distance;
}

double SVRParameters::get_svr_adjacent_levels_ratio() const
{
    return svr_adjacent_levels_ratio;
}


void SVRParameters::set_svr_adjacent_levels_ratio(const double _svr_adjacent_levels_ratio)
{
    if (_svr_adjacent_levels_ratio < 0. || _svr_adjacent_levels_ratio > 1.)
        THROW_EX_FS(std::range_error, "Adjacent levels ratio " << _svr_adjacent_levels_ratio << " is out of 0..1 range.");
    svr_adjacent_levels_ratio = _svr_adjacent_levels_ratio;
}


kernel_type_e SVRParameters::get_kernel_type() const
{
    return kernel_type;
}

bool SVRParameters::is_manifold() const
{
    return kernel_type == kernel_type_e::DEEP_PATH || kernel_type == kernel_type_e::DEEP_PATH2;
}

void SVRParameters::set_kernel_type(const kernel_type_e _kernel_type)
{
    if (_kernel_type < kernel_type_e::number_of_kernel_types)
        kernel_type = _kernel_type;
    else
        THROW_EX_FS(std::invalid_argument, "Wrong kernel type " << (ssize_t) _kernel_type);
}

size_t SVRParameters::get_lag_count() const
{
    return lag_count;
}

void SVRParameters::set_lag_count(const size_t _lag_count)
{
    if (_lag_count == 0) THROW_EX_FS(std::invalid_argument, "Lag count parameter cannot be zero.");
    lag_count = _lag_count;
}


std::string SVRParameters::to_string() const
{
    std::stringstream s;
    s << std::setprecision(std::numeric_limits<double>::max_digits10);
    s << "id " << id
      << ", dataset id " << dataset_id
      << ", cost " << svr_C
      << ", epsilon " << svr_epsilon
      << ", kernel param " << svr_kernel_param
      << ", kernel param 2 " << svr_kernel_param2
      << ", decrement distance " << svr_decremental_distance
      << ", svr adjacent levels ratio " << svr_adjacent_levels_ratio
      << ", kernel type " << static_cast<int>(kernel_type)
      << ", lag count " << lag_count
      << ", table name " << input_queue_table_name
      << ", column name " << input_queue_column_name
      << ", decon level " << decon_level_
      << ", chunk " << chunk_ix_
      << ", gradient " << grad_level_;

    return s.str();
}

std::string SVRParameters::to_sql_string() const
{
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);

    s << "\t" << get_id()
      << "\t" << get_dataset_id()
      << "\t" << input_queue_table_name
      << "\t" << input_queue_column_name
      << "\t" << decon_level_
      << "\t" << chunk_ix_
      << "\t" << grad_level_
      << "\t" << svr_C
      << "\t" << svr_epsilon
      << "\t" << svr_kernel_param
      << "\t" << svr_kernel_param2
      << "\t" << svr_decremental_distance
      << "\t" << svr_adjacent_levels_ratio
      << "\t" << static_cast<int>(kernel_type)
      << "\t" << lag_count;

    return s.str();
}

bool SVRParameters::from_sql_string(const std::string &sql_string)
{
    std::vector<std::string> tokens;
    svr::common::split(sql_string, '\t', tokens);

    if (tokens.size() != 13) {
        LOG4_ERROR("Incorrect number of tokens " << tokens.size());
        return false;
    }
    set_id(atoll(tokens[0].c_str()));
    set_dataset_id(atoll(tokens[1].c_str()));
    set_input_queue_table_name(tokens[2]);
    set_input_queue_column_name(tokens[3]);
    set_decon_level(atol(tokens[4].c_str()));
    set_chunk_ix(atol(tokens[5].c_str()));
    set_grad_level(atol(tokens[6].c_str()));
    set_svr_C(atof(tokens[7].c_str()));
    set_svr_epsilon(atof(tokens[8].c_str()));
    set_svr_kernel_param(atof(tokens[9].c_str()));
    set_svr_kernel_param2(atof(tokens[10].c_str()));
    set_svr_decremental_distance(atoll(tokens[11].c_str()));
    set_svr_adjacent_levels_ratio(atof(tokens[12].c_str()));
    set_kernel_type(kernel_type_e(atol(tokens[13].c_str())));
    set_lag_count(atoll(tokens[14].c_str()));

    LOG4_DEBUG("Successfully loaded parameters " << to_sql_string());

    return true;
}

}
}