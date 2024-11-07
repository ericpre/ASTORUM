#ifndef EDXSDATASET_H
#define EDXSDATASET_H

#include "modelutils.h"

class EDXSDataset {
    public:
        int _Gcols;
        int _beamEnergy;
        EDXSModelConstants::problemType _problemType;
        EDXSModelConstants::absorptionModelType _absorptionModelType;
        double _azimuthAngle;
        double _elevationAngle;
        double _tiltStage;
        double _thickness;
        double _density;
        double _widthSlope;
        double _widthIntercept;
        double _energyAxisSize;
        double _energyAxisScale;
        double _energyAxisOffset;
        std::string _detectorEfficiency;
        std::string _xrayDB;
        std::string _massAbsorptionCoefficientsFilePath;
        std::string _decompositionResultsFilePath;
        std::string _absorptionMatrixFilePath;
        std::string _thicknessMapFilePath;
        std::string _periodicTableInfoFilePath;
        std::vector<std::string> _elements;
        std::vector<std::string> _splitLinesElements;
        std::vector<double> _energyThresholds;
        std::vector<std::string> _modelElements;
        std::vector<std::string> _quantificationElements;
        std::vector<std::string> _absorptionElements;
        Eigen::VectorXd _absorptionElementsConcentrations;
        Eigen::VectorXd _energyScale;
        Eigen::MatrixXd _X;
        Eigen::VectorXd _norms;
        Eigen::MatrixXd _detectorEfficiencyMatrix;
        Eigen::MatrixXd _G;
        Eigen::VectorXd _T;
        Eigen::MatrixXd _A;
        nlohmann::json _xrayDBFile;
        nlohmann::json _massAbsorptionCoefficientsDBFile;
        nlohmann::json _periodicTableInfoDBFile;

    public:
        EDXSDataset();

        EDXSDataset(
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

        ~EDXSDataset();

        void generateElementalGMatrix();

        void generateBremsstrahlungGMatrix();

        void generateGMatrix();

        std::vector<int> NMFSimplexIndices();

        std::vector<std::string> NMFSimplexElements();

        std::vector<int> selectedElementsIndices(const std::vector<std::string>& selectedElements);

        void updateGBremsstrahlung(const Eigen::Ref<const Eigen::MatrixXd>& W);

        Eigen::VectorXd readThicknessMap();

        Eigen::MatrixXd generateQuantificationMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H);

        Eigen::MatrixXd generateQuantificationMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, const std::vector<std::string>& selectedElements);

        Eigen::VectorXd computeDensityMap(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H);

        Eigen::VectorXd computeDensityMap(const Eigen::Ref<const Eigen::MatrixXd>& Q, const std::vector<std::string>& elements);
        
        Eigen::MatrixXd generateAbsorptionCorrectionMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H);

        Eigen::MatrixXd generateAbsorptionCorrectionMatrix(const Eigen::Ref<const Eigen::MatrixXd>& Q, const std::vector<std::string>& elements);

        void applyAbsorptionCorrection(Eigen::Ref<Eigen::MatrixXd> X);
}; 

#endif