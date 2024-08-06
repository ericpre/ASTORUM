#ifndef MODELUTILS_H
#define MODELUTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <map>
#include <vector>
#include <unordered_set>
#include <regex>
#include <omp.h>
#include "constants.hpp"

void writeMatrixToFile(const Eigen::Ref<const Eigen::MatrixXd>& A, const std::string& filename);

void writeVectorToFile(const Eigen::Ref<const Eigen::VectorXd>& v, const std::string& filename);

Eigen::MatrixXd readMatrixFromFile(const std::string& filename);

Eigen::VectorXd readVectorFromFile(const std::string& filename);

Eigen::VectorXd buildEnergyScale(double energyAxisSize, double energyAxisScale, double energyAxisOffset);

void gaussian(const Eigen::Ref<const Eigen::VectorXd>& x, double mu, double sigma, Eigen::Ref<Eigen::VectorXd> f);

int countGColumns(const std::vector<std::string>& elements, const std::vector<std::string>& splitLinesElements);

Eigen::MatrixXd readXrayLines(const std::string& element, const nlohmann::json& xrayDBFile);

Eigen::MatrixXd readMassAbsorptionCoefficients(const std::string& element, const nlohmann::json& massAbsorptionCoefficientFile);

Eigen::MatrixXd readDetectorEfficiency(const std::string& filename);

int findClosestValueIndexInVector(const Eigen::Ref<const Eigen::VectorXd>& v, double value);

Eigen::VectorXd lifshinBremsstrahlungB0(const Eigen::Ref<const Eigen::VectorXd>& x, double b0, double E0);

Eigen::VectorXd lifshinBremsstrahlungB1(const Eigen::Ref<const Eigen::VectorXd>& x, double b1, double E0);

Eigen::VectorXd atomicToWeightPercent(const Eigen::Ref<const Eigen::VectorXd>& atomicPercentages, const std::vector<std::string>& elements, const nlohmann::json& periodicTableInfoFile);

double densityOfMixture(const Eigen::Ref<const Eigen::VectorXd>& weightPercentages, const std::vector<std::string>& elements, const nlohmann::json& periodicTableInfoFile, EDXSModelConstants::meanType meanType);

int energyToArrayIndex(double energy, double energyAxisScale, double energyAxisOffset);

double _basic_simpson(const Eigen::Ref<const Eigen::VectorXd>& y, int start, int stop, const Eigen::Ref<const Eigen::VectorXd>& x, double dx = 1.0);

double simpson(const Eigen::Ref<const Eigen::VectorXd>& y, const Eigen::Ref<const Eigen::VectorXd>& x, double dx = 1.0);

double F(double electronEnergy);

double thetaE(double density, double electronEnergy);

double electronInelasticMeanFreePath(double density, double electronEnergy);

double angularCorrection(double density, double electronEnergy, double alpha, double beta);

double estimateThicknessAtPixel(const Eigen::Ref<const Eigen::VectorXd>& EELSLowLossSpectrum, const Eigen::Ref<const Eigen::VectorXd>& energyAxis, double energyAxisScale, double energyAxisOffset, double zeroLossPeakThreshold, double density, double electronEnergy, double alpha, double beta);

Eigen::VectorXd computeMassAbsorptionCoefficients(const Eigen::Ref<const Eigen::VectorXd>& energyRange, const std::vector<std::string>& elements, const Eigen::Ref<const Eigen::VectorXd>& concentrations, bool atomicFraction, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile);

double computeSingleMassAbsorptionCoefficient(double energy, const std::string& element, const nlohmann::json& massAbsorptionCoefficientFile);

Eigen::VectorXd computeAbsorptionCorrection(const Eigen::Ref<const Eigen::VectorXd>& energyRange, const std::vector<std::string>& elements, const Eigen::Ref<const Eigen::VectorXd>& concentrations, double thickness, double takeOffAngle, double* density, bool atomicFraction, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile);

double computeSingleAbsorptionCorrection(double energy, const std::string& element, double thickness, double takeOffAngle, double* density, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile);

Eigen::MatrixXd getExplainedIntensity(const Eigen::Ref<const Eigen::MatrixXd>& G, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H);

void printConcentrationReport(const Eigen::Ref<const Eigen::MatrixXd>& G, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, const std::vector<std::string>& modelElements, const std::vector<std::string>& selectedElements, bool fitError);

#endif