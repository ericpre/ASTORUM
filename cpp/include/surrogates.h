#ifndef SURROGATES_H
#define SURROGATES_H

#include "measures.h"
#include "constants.hpp"

double smoothL2Surrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, double sigmaL, double lambdaL);

double smoothDGKLSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, double sigmaL, double lambdaL);

double diffSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, double sigmaL, double lambdaL, SmoothNMFConstants::algorithm algorithm);

double quadraticSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& A, const Eigen::Ref<const Eigen::MatrixXd>& Aold, const Eigen::Ref<const Eigen::MatrixXd>& grad, double lossOld, double gammaVal);

#endif