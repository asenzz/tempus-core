#pragma once
#include<string>
#include<vector>
#include"../Runner/hdbscanRunner.hpp"
#include"../Runner/hdbscanParameters.hpp"
#include"../Runner/hdbscanResult.hpp"
#include"../HdbscanStar/outlierScore.hpp"

using namespace std;


class Hdbscan

{
private:
    string fileName;

    hdbscanResult result;

public:
    vector<vector<double> > dataset;

    std::vector<int> labels_;

    std::vector<int> normalizedLabels_;

    std::vector<outlierScore> outlierScores_;

    std::vector<double> membershipProbabilities_;

    uint32_t noisyPoints_;

    uint32_t numClusters_;


    Hdbscan(string readFileName)
    {
        fileName = readFileName;
    }

    Hdbscan(const std::vector<std::vector<double> > &data) : dataset(data)
    {
    }

    string getFileName();

    int loadCsv(int numberOfValues, bool skipHeader = false);

    void execute(const int minPoints, const int minClusterSize, const std::string &distanceMetric);

    void displayResult();
};
