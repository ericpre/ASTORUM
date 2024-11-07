#ifndef MEASURES_H
#define MEASURES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <stdexcept>

double FrobeniusLoss(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, bool average);

double KLDivLoss(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double logShift, bool average);

double traceXTX(const Eigen::Ref<const Eigen::MatrixXd>& HT, bool average);

double logReg(const Eigen::Ref<const Eigen::MatrixXd>& H, double mu, double epsilonReg, bool average);

#endif