#ifndef BLOCKWISESMOOTHNMF_H
#define BLOCKWISESMOOTHNMF_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <map>
#include <iterator>
#include <unordered_set>
#include <Eigen/Dense>
#include "SmoothNMF.h"

double euclideanDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

double manhattanDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

double mahalanobisDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b, const Eigen::MatrixXd& inverseCovarianceMatrix);

double minkowskiDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b, int p);

double cosineDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

double chebyshevDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

double hammingDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

double jaccardDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b);

Eigen::VectorXd NNLS(const Eigen::Ref<const Eigen::MatrixXd>& M, const Eigen::Ref<const Eigen::VectorXd>& y);


class BlockWiseSmoothNMF {
    public:
        std::string _inputDir;
        std::string _outputDir;

        int _blocksHorizontal;
        int _blocksVertical;
        int _blocks;
        int _pixels;
        int _blockWidth;
        int _blockHeight;
        int _Gcols;
        int _nPoints;
        int _nDims;
        int _nClusters;

        Eigen::VectorXi _componentsVector;

        Eigen::MatrixXd _WElements;
        Eigen::MatrixXd _WBremsstrahlung;
        Eigen::MatrixXd _WFull;
        Eigen::MatrixXd _WClustered;

        std::vector<std::vector<int>> _clusterLabels;
        Eigen::MatrixXi _componentPresences;

        EDXSDataset _model;
        Eigen::MatrixXd _G;
        Eigen::MatrixXd _GB0;
        Eigen::MatrixXd _GB1;
    
    public:
        BlockWiseSmoothNMF();

        BlockWiseSmoothNMF(
            const std::string& inputDir,
            const std::string& outputDir,
            int blocksHorizontal,
            int blocksVertical,
            int blockWidth,
            int blockHeight,
            int nClusters,
            const Eigen::Ref<const Eigen::VectorXi>& componentsVector
            );

        ~BlockWiseSmoothNMF();

        void initialiseModel(
            int beamEnergy,
            EDXSModelConstants::problemType problemType,
            EDXSModelConstants::absorptionModelType absorptionModelType,
            double azimuthAngle,
            double elevationAngle,
            double tiltStage,
            double thickness,
            double density,
            double widthSlope,
            double widthIntercept,
            double energyAxisSize,
            double energyAxisScale,
            double energyAxisOffset,
            std::string detectorEfficiency,
            std::string xrayDB,
            std::string massAbsorptionCoefficientsFilePath,
            std::string decompositionResultsFilePath,
            std::string absorptionMatrixFilePath,
            std::string thicknessMapFilePath,
            std::string periodicTableInfoFilePath,
            std::vector<std::string> elements,
            std::vector<std::string> splitLinesElements,
            std::vector<double> energyThresholds,
            std::vector<std::string> quantificationElements,
            std::vector<std::string> absorptionElements,
            Eigen::VectorXd absorptionElementsConcentrations
        );

        void setModel(EDXSDataset model);

        Eigen::MatrixXd postProcessData(const Eigen::Ref<const Eigen::MatrixXd>& W, int separationOrder);

        void computeWBlocks(  
            const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            SmoothNMFConstants::algorithm algorithm,
            double tol, 
            double logShift, 
            double eps, 
            double lambdaL, 
            double mu, 
            double epsilonReg, 
            double dichotomyTol, 
            double sigmaL, 
            double gammaStepScalar,
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe,
            bool debug, 
            bool normalise, 
            bool noStopCriterion, 
            bool lineSearch,
            const Eigen::Ref<const Eigen::VectorXi>& separationOrder,
            bool precomputedW,
            bool writeWblocks
            );

        void KmeansClustering(const Eigen::Ref<const Eigen::MatrixXd>& data, int nIter, double tolerance);

        void spectralClustering(double sigma, BlockWiseSmoothNMFConstants::distanceMetric metric, int p, int nIter, double tolerance);

        double estimateSNR(const Eigen::Ref<const Eigen::MatrixXd>& R, const Eigen::Ref<const Eigen::VectorXd>& r_m, const Eigen::Ref<Eigen::MatrixXd>& x);

        void VCA(bool verbose);

        Eigen::MatrixXd computeAbundanceVCA();

        void printClusterLabels();

        void computeClusteredW();

        void computeComponentPresences();

        void computeComponentPresencesVCA();

        void refineW(  
            const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            SmoothNMFConstants::algorithm algorithm,
            double tol, 
            double logShift, 
            double eps, 
            double lambdaL, 
            double mu, 
            double epsilonReg, 
            double dichotomyTol, 
            double sigmaL, 
            double gammaStepScalar,
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe,
            bool debug, 
            bool normalise, 
            bool noStopCriterion, 
            bool lineSearch
            );

        void refineWVCA(  
            const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            SmoothNMFConstants::algorithm algorithm,
            double tol, 
            double logShift, 
            double eps, 
            double lambdaL, 
            double mu, 
            double epsilonReg, 
            double dichotomyTol, 
            double sigmaL, 
            double gammaStepScalar,
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe,
            bool debug, 
            bool normalise, 
            bool noStopCriterion, 
            bool lineSearch
            );

        Eigen::MatrixXd getMonolithicX();

        void computeHBlocks( 
            const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            SmoothNMFConstants::algorithm algorithm,
            double tol, 
            double logShift, 
            double eps, 
            double lambdaL, 
            double mu, 
            double epsilonReg, 
            double dichotomyTol, 
            double sigmaL, 
            double gammaStepScalar,
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe,
            bool debug, 
            bool normalise, 
            bool noStopCriterion, 
            bool lineSearch
        );

        void computeHMatrix( 
            const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            SmoothNMFConstants::algorithm algorithm,
            double tol, 
            double logShift, 
            double eps, 
            double lambdaL, 
            double mu, 
            double epsilonReg, 
            double dichotomyTol, 
            double sigmaL, 
            double gammaStepScalar,
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe,
            bool debug, 
            bool normalise, 
            bool noStopCriterion, 
            bool lineSearch
        );

        void computeHBlocksSVD();

        void computeHMatrixSVD();

        void computeABlocks(BlockWiseSmoothNMFConstants::fusionType fusionType);

        void computeAMatrix(BlockWiseSmoothNMFConstants::fusionType fusionType);
};

#endif