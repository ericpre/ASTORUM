#ifndef SMOOTHNMF_H
#define SMOOTHNMF_H

#include "updates.h"
#include "surrogates.h"
#include "measures.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <Eigen/unsupported/NNLS>

void displayProgressBar(int currentIteration, int maxIterations, double loss, double elapsedSeconds);

void rescaleDH(Eigen::Ref<Eigen::MatrixXd> D, Eigen::Ref<Eigen::MatrixXd> H); 

class NMFEstimator {
    public:
        int _channels;
        int _pixels;
        int _components;
        int _Gcols;
        SmoothNMFConstants::initialisation _init;
        int _maxIter;
        int _randomSeed;
        int _nIter;

        double _tol;
        double _logShift;
        double _normFactor;
        double _constKL;
        double _eps;
        double _detailedLoss;
        double _reconstructionLoss;

        bool _simplexW;
        bool _simplexH;
        bool _l2;
        bool _verbose;
        bool _safe;
        bool _debug;
        bool _normalise;
        bool _noStopCriterion;

        Eigen::MatrixXd _X;
        Eigen::MatrixXd _W;
        Eigen::MatrixXd _H;
        Eigen::MatrixXd _G;
        Eigen::MatrixXd _fixedW;
        Eigen::MatrixXd _fixedH;
        Eigen::SparseMatrix<double> _L;

        EDXSDataset _model;

    public:
        NMFEstimator();

        NMFEstimator(
            const Eigen::Ref<const Eigen::MatrixXd>& X, 
            double logShift
            );

        NMFEstimator(
            const Eigen::Ref<const Eigen::MatrixXd>& X, 
            const Eigen::Ref<const Eigen::MatrixXd>& W, 
            const Eigen::Ref<const Eigen::MatrixXd>& H, 
            const Eigen::Ref<const Eigen::MatrixXd>& G, 
            const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
            const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
            int components, 
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            double tol, 
            double logShift, 
            double eps, 
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe, 
            bool debug, 
            bool normalise, 
            bool noStopCriterion
            );

        NMFEstimator(
            const Eigen::Ref<const Eigen::MatrixXd>& X, 
            const Eigen::Ref<const Eigen::MatrixXd>& W, 
            const Eigen::Ref<const Eigen::MatrixXd>& H,
            const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
            const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
            int components, 
            SmoothNMFConstants::initialisation init, 
            int maxIter, 
            int randomSeed, 
            double tol, 
            double logShift, 
            double eps, 
            bool simplexW, 
            bool simplexH, 
            bool l2, 
            bool verbose, 
            bool safe, 
            bool debug, 
            bool normalise, 
            bool noStopCriterion
            );

        ~NMFEstimator();

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

        double normalisationFactor(const Eigen::Ref<const Eigen::MatrixXd>& X, double f);
        void removeZeroLines(Eigen::Ref<Eigen::MatrixXd> X, double eps);
        double loss(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, bool average);

        virtual void iteration();

        void fitTransform();
    };

    class SmoothNMF : public NMFEstimator {
        public:
            Eigen::VectorXd _gammaStepArray;

            SmoothNMFConstants::algorithm _algorithm;

            double _lambdaL;
            double _mu;
            double _epsilonReg;
            double _dichotomyTol;
            double _sigmaL;
            double _gammaStepScalar;

            bool _lineSearch;

        public:
            SmoothNMF();

            SmoothNMF(
                const Eigen::Ref<const Eigen::MatrixXd>& X, 
                double logShift
                );

            SmoothNMF(
                const Eigen::Ref<const Eigen::MatrixXd>& X, 
                const Eigen::Ref<const Eigen::MatrixXd>& W, 
                const Eigen::Ref<const Eigen::MatrixXd>& H, 
                const Eigen::Ref<const Eigen::MatrixXd>& G, 
                const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
                const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
                const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
                int components, 
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

            SmoothNMF(
                const Eigen::Ref<const Eigen::MatrixXd>& X, 
                const Eigen::Ref<const Eigen::MatrixXd>& W, 
                const Eigen::Ref<const Eigen::MatrixXd>& H,
                const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
                const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
                const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
                int components, 
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

            ~SmoothNMF();

            double lossSmoothNMF(Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, bool average);
            
            void iteration();

            void fitTransform();
    };

#endif