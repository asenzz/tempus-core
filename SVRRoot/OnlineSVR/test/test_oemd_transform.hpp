//
// Created by sstoyanov on 3/15/18.
//

#ifndef ONLINE_EMD_TEST_OEMD_TRANSFORM_HPP
#define ONLINE_EMD_TEST_OEMD_TRANSFORM_HPP

#include <memory>
#include <fstream>



std::ifstream open_data_file(std::string file_name)
{

    std::ifstream f(file_name);
    if(!f)
        throw std::runtime_error("Cannot find the input data file");

    return f;
}

std::vector<double> read_test_data(const std::string &file_name_to_search)
{
    auto f = open_data_file(file_name_to_search);

    std::vector<double> result;

    while(f.good())
    {
        double a;
        f >> a;
        result.push_back(a);
    }
    return result;
}




#endif //ONLINE_EMD_TEST_OEMD_TRANSFORM_HPP
