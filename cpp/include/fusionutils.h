#ifndef FUSIONUTILS_H
#define FUSIONUTILS_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <omp.h>

Eigen::VectorXd estimateHAADF(const Eigen::Ref<const Eigen::MatrixXd>& Q, const Eigen::Ref<const Eigen::VectorXi>& Z, double gamma);

Eigen::MatrixXd computeResidualGradient(const Eigen::Ref<const Eigen::VectorXd>& R, const Eigen::Ref<const Eigen::VectorXi>& Z);

double forwardModelCost(const Eigen::Ref<const Eigen::VectorXd> HAADFInit, const Eigen::Ref<const Eigen::MatrixXd>& Q, const Eigen::Ref<const Eigen::VectorXi>& Z, double gamma);

double poissonTermCost(const Eigen::Ref<const Eigen::MatrixXd>& QInit, const Eigen::Ref<const Eigen::MatrixXd>& Q, double epsilon);

void TVObjective2D(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input, Eigen::Ref<Eigen::MatrixXd> Output, const Eigen::Ref<const Eigen::MatrixXd>& Px, const Eigen::Ref<const Eigen::MatrixXd>& Py, double lambda);

void TVGradient2D(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input, Eigen::Ref<Eigen::MatrixXd> Px, Eigen::Ref<Eigen::MatrixXd> Py, double lFactor);

void TVProject2D(int xDim, int yDim, Eigen::Ref<Eigen::MatrixXd> Px, Eigen::Ref<Eigen::MatrixXd> Py);

Eigen::MatrixXd TVFastGradientProjection(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input, double lambdaReg, int nIter);

double TVRegularisationCost(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input);

#endif