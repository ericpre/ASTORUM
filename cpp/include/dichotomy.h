#ifndef DICHOTOMY_H
#define DICHOTOMY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <stdexcept>

void functionSimplex(const Eigen::Ref<const Eigen::VectorXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& num, const Eigen::Ref<const Eigen::MatrixXd>& denum, double logShift, Eigen::Ref<Eigen::VectorXd> R);

void dichotomySimplex(const Eigen::Ref<const Eigen::MatrixXd>& A, const Eigen::Ref<const Eigen::MatrixXd>& B, double logShift, double tol, int maxIter, Eigen::Ref<Eigen::VectorXd> R);

void functionSimplexProjectedGradient(const Eigen::Ref<const Eigen::VectorXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& A, double logShift, Eigen::Ref<Eigen::VectorXd> R);

void dichotomySimplexProjectedGradient(const Eigen::Ref<const Eigen::MatrixXd>& A, double logShift, double tol, int maxIter, Eigen::Ref<Eigen::VectorXd> R);

#endif