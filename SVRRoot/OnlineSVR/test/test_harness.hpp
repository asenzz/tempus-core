#ifndef TEST_HARNESS_HPP
#define TEST_HARNESS_HPP

#include <string>
#include <vector>
#include <ostream>

#include <util/math_utils.hpp>

bool erase_after(std::string &where, char what);

std::string exec(char const *cmd, int &procExitCode);

std::string exec(std::string const &cmd);

std::vector<double> read_test_data(const std::string &file_name_to_search);

std::ifstream open_data_file(std::string file_name_to_search);

std::vector<double> read_length_of_test_data(std::string file_name_to_search, int length);

template<class T>
struct delimiters
{
    static constexpr char const *delim = " ";
    static constexpr char const *opener = "";
    static constexpr char const *closer = "";
};


template<class T>
struct delimiters<std::vector<T>>
{
    static constexpr char const *delim = " ";
    static constexpr char const *opener = "";
    static constexpr char const *closer = "";
};


template<class T>
struct delimiters<std::vector<std::vector<T>>>
{
    static constexpr char const *delim = "\n";
    static constexpr char const *opener = "";
    static constexpr char const *closer = "";
};


template<class T>
typename std::enable_if<std::is_pod<T>::value, void>::type
pretty_print(std::basic_ostream<char> &ostr, T t, char const *)
{
    ostr << t;
}

template<class Cont>
typename std::enable_if<!std::is_pod<Cont>::value, void>::type
pretty_print(std::basic_ostream<char> &ostr, Cont const &cont, char const *delim = delimiters<Cont>::delim)
{
    auto beg = std::begin(cont);
    auto const end = std::end(cont);

    ostr << delimiters<DTYPE(*beg)>::opener;

    if (beg != end) {
        pretty_print(ostr, *beg, delimiters<DTYPE(*beg)>::delim);
        while (++beg != end) {
            ostr << delim;
            pretty_print(ostr, *beg, delimiters<DTYPE(*beg)>::delim);
        }
    }
    ostr << delimiters<DTYPE(*beg)>::closer;
}

#define ASSERT_FP_EQ(op1, op2) ASSERT_TRUE(svr::common::equals(op1, op2)) << " op1: " << (op1) << " op2: " << (op2)


#endif /* TEST_HARNESS_HPP */

