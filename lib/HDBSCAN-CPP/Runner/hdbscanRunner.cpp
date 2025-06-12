#include <functional>
#include "hdbscanRunner.hpp"
#include "hdbscanResult.hpp"
#include "hdbscanParameters.hpp"
#include"../Distance/EuclideanDistance.hpp"
#include"../Distance/ManhattanDistance.hpp"
#include"../HdbscanStar/hdbscanAlgorithm.hpp"
#include"../HdbscanStar/undirectedGraph.hpp"
#include"../HdbscanStar/cluster.hpp"
#include"../HdbscanStar/outlierScore.hpp"

using namespace hdbscanStar;

enum class e_distance_fun
{
    euclidean,
    manhattan
};

hdbscanResult hdbscanRunner::run(hdbscanParameters parameters)
{
    int numPoints = parameters.dataset.size() != 0 ? parameters.dataset.size() : parameters.distances.size();

    hdbscanAlgorithm algorithm;
    hdbscanResult result;
    if (parameters.distances.size() == 0) {
        std::vector<std::vector<double> > distances(numPoints);
        e_distance_fun distance_fun;
        if (parameters.distanceFunction == "Manhattan") distance_fun = e_distance_fun::manhattan;
        else distance_fun = e_distance_fun::euclidean;
#pragma omp parallel for
        for (int i = 0; i < numPoints; ++i) {
            distances[i].resize(numPoints);
            for (int j = 0; j < i; ++j) {
                switch (distance_fun) {
                    case e_distance_fun::manhattan: {
                        ManhattanDistance MDistance;
                        const auto distance = MDistance.computeDistance(parameters.dataset[i], parameters.dataset[j]);
                        distances[i][j] = distance;
                        distances[j][i] = distance;
                        break;
                    }
                    default: {
                        //Default to Euclidean
                        EuclideanDistance EDistance;
                        const auto distance = EDistance.computeDistance(parameters.dataset[i], parameters.dataset[j]);
                        distances[i][j] = distance;
                        distances[j][i] = distance;
                    }
                }
            }
        }

        parameters.distances = distances;
    }

    const auto coreDistances = algorithm.calculateCoreDistances(parameters.distances, parameters.minPoints);

    undirectedGraph mst = algorithm.constructMst(parameters.distances, coreDistances, true);
    mst.quicksortByEdgeWeight();

    std::vector<double> pointNoiseLevels(numPoints);
    std::vector<int> pointLastClusters(numPoints);

    std::vector<std::vector<int> > hierarchy;

    std::vector<cluster *> clusters;
    algorithm.computeHierarchyAndClusterTree(
        &mst,
        parameters.minClusterSize,
        parameters.constraints,
        hierarchy,
        pointNoiseLevels,
        pointLastClusters,
        clusters);
    bool infiniteStability = algorithm.propagateTree(clusters);

    const auto prominentClusters = algorithm.findProminentClusters(clusters, hierarchy, numPoints);
    const auto membershipProbabilities = algorithm.findMembershipScore(prominentClusters, coreDistances);
    const auto scores = algorithm.calculateOutlierScores(
        clusters,
        pointNoiseLevels,
        pointLastClusters,
        coreDistances);

    return hdbscanResult(prominentClusters, scores, membershipProbabilities, infiniteStability);
}
