#include "ManhattanDistance.hpp"
#include<vector>
#include<cmath>
#include<cstdint>

double ManhattanDistance::computeDistance(const std::vector<double> &attributesOne, const std::vector<double> &attributesTwo)
{
    double distance = 0;
    for (uint32_t i = 0; i < attributesOne.size() && i < attributesTwo.size(); ++i) distance += fabs(attributesOne[i] - attributesTwo[i]);
    return distance;
}
