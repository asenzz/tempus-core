#pragma once
#include"IDistanceCalculator.hpp"

/// <summary>
/// Computes the manhattan distance between two points, d = |x1-y1| + |x2-y2| + ... + |xn-yn|.
/// </summary>
class ManhattanDistance : IDistanceCalculator
{
public:
    double computeDistance(const std::vector<double> &attributesOne, const std::vector<double> &attributesTwo);
};
