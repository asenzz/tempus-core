#pragma once

#include <exception>
#include <armadillo>

namespace svr {
namespace common {

namespace color {
enum e_code
{
    FG_RED = 31,
    FG_GREEN = 32,
    FG_BLUE = 34,
    FG_DEFAULT = 39,
    BG_RED = 41,
    BG_GREEN = 42,
    BG_BLUE = 44,
    BG_DEFAULT = 49
};

class modifier
{
    e_code code;
public:
    modifier(e_code pCode) : code(pCode)
    {}

    friend std::ostream &
    operator<<(std::ostream &os, const modifier &mod)
    {
        return os << "\033[" << mod.code << "m";
    }
};
}

class invalid_data : public std::logic_error
{
public:
    invalid_data(const std::string &msg) : std::logic_error(std::string("svr::common::invalid_data: ") + msg)
    {}

};

class insufficient_data : public std::range_error
{
public:
    insufficient_data(const std::string &msg) : std::range_error(std::string("svr::common::insufficient_data: ") + msg)
    {}

};

class invalid_request : public std::logic_error
{
public:
    invalid_request(const std::string &msg) : std::logic_error(std::string("svr::common::invalid_request: ") + msg)
    {}
};

class corrupted_model : public std::runtime_error
{
public:
    corrupted_model(const std::string &msg) : std::runtime_error(std::string("svr::common::corrupted_model: ") + msg)
    {}
};

class missing_data : public std::logic_error
{
private:
    bpt::ptime last_available;
public:
    missing_data(const std::string &msg) : std::logic_error(std::string("svr::common::missing_data: ") + msg)
    {}

    missing_data(const std::string &msg, bpt::ptime _last_available) : std::logic_error(std::string("svr::common::missing_data: ") + msg), last_available(_last_available)
    {}

    bpt::ptime get_last_available() const
    {
        return this->last_available;
    }
};

class missing_data_fatal : public std::logic_error
{
private:
public:
    missing_data_fatal(const std::string &msg) : std::logic_error(std::string("svr::common::missing_data_fatal: ") + msg)
    {}
};


class bad_model : public std::logic_error
{
private:
    size_t decon_level;
    std::string column_name;
public:
    bad_model(const std::string &msg) : std::logic_error(std::string("svr::common::bad_model: ") + msg)
    {}

    bad_model(const std::string &msg, const size_t decon_level_, const std::string column_name_) : std::logic_error(std::string("svr::common::bad_model: ") + msg),
                                                                                                   decon_level(decon_level_), column_name(column_name_)
    {}

    std::string get_column_name() const
    {
        return column_name;
    }

    size_t get_decon_level() const
    {
        return decon_level;
    }
};

class bad_prediction : public bad_model
{
private:
    arma::mat predicted;
public:
    bad_prediction(const std::string &msg) : bad_model(std::string("svr::common::bad_prediction: ") + msg)
    {}

    bad_prediction(const std::string &msg, const arma::mat &predicted) : bad_model(std::string("svr::common::bad_prediction: ") + msg)
    {}

    bad_prediction(const std::string &msg, const size_t decon_level_, const std::string column_name_) : bad_model(std::string("svr::common::bad_prediction: ") + msg,
                                                                                                                  decon_level_, column_name_)
    {}

    bad_prediction(const std::string &msg, const arma::mat &predicted_, const size_t decon_level_, const std::string column_name_) : bad_model(
            std::string("svr::common::bad_prediction: ") + msg, decon_level_, column_name_), predicted(predicted_)
    {}

};

class control_outlier : public bad_model
{

public:
    control_outlier(const std::string &msg) : bad_model(std::string("svr::common::control_outlier: ") + msg)
    {}

    control_outlier(const std::string &msg, const size_t decon_level_, const std::string column_name_) : bad_model(std::string("svr::common::control_outlier: ") + msg,
                                                                                                                   decon_level_, column_name_)
    {}
};

class chunk_outlier : public bad_model
{

public:
    chunk_outlier(const std::string &msg) : bad_model(std::string("svr::common::chunk_outlier: ") + msg)
    {}

    chunk_outlier(const std::string &msg, const size_t decon_level_, const std::string column_name_) : bad_model(std::string("svr::common::chunk_outlier: ") + msg,
                                                                                                                 decon_level_, column_name_)
    {}
};


}
}
