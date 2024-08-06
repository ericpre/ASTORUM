#ifndef UPDATES_H
#define UPDATES_H

#include "dichotomy.h"
#include "EDXSDataset.h"
#include "constants.hpp"
#include <random>

void initialiseNMF(const Eigen::Ref<const Eigen::MatrixXd>& X, int components, SmoothNMFConstants::initialisation init, double epsilon, int randSeed, bool debug, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H);

void initialiseAlgorithms(const Eigen::Ref<const Eigen::MatrixXd>& X, int components, SmoothNMFConstants::initialisation init, double epsilon, int randSeed, bool simplexW, bool simplexH, bool debug, double logShift, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<Eigen::MatrixXd> G, EDXSDataset& model);

void multiplicativeUpdateW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> H, bool simplexW, double logShift, bool safe, bool debug, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedW, bool useBregman, Eigen::Ref<Eigen::MatrixXd> W, EDXSDataset& model);

void multiplicativeUpdateH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, bool simplexH, double mu, double epsilonReg, double lambdaL, double logShift, bool safe, bool debug, double dichotomyTol, double sigmaL, bool l2, bool useBregman, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, const Eigen::Ref<const Eigen::MatrixXd>& fixedH, Eigen::Ref<Eigen::MatrixXd> H);

void gradW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double logShift, bool safe, bool debug, bool l2, Eigen::Ref<Eigen::MatrixXd> grad);

void gradH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double mu, double lambdaL, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, double epsilonReg, double logShift, bool safe, bool debug, bool l2, Eigen::Ref<Eigen::MatrixXd> grad);

double estimateLipschitzBoundW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, int k, double logShift, bool debug);

double estimateLipschitzBoundH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, int k, double logShift, double lambdaL, double mu, double epsilonReg, bool debug);

void projectedGradientStepW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> H, double gamma, bool simplexW, double logShift, bool safe, bool debug, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedW, Eigen::Ref<Eigen::MatrixXd> W);

void projectedGradientStepH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, double gamma, bool simplexH, double mu, double logShift, double epsilonReg, bool safe, bool debug, double dichotomyTol, double lambdaL, const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedH, Eigen::Ref<Eigen::MatrixXd> H);

#endif