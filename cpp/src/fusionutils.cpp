#include "../include/fusionutils.h"

Eigen::VectorXd estimateHAADF(const Eigen::Ref<const Eigen::MatrixXd>& Q, const Eigen::Ref<const Eigen::VectorXi>& Z, double gamma) {
    int pixels = Q.cols();
    int elements = Z.size();

    if (Q.rows() != elements) {
        throw std::invalid_argument("Number of quantification maps must match the number of specified elements.");
    }

    Eigen::VectorXd HAADF = Eigen::VectorXd::Zero(pixels);
    double zMean = Z.mean();

    #pragma omp parallel for
    for (int i = 0; i < pixels; i++) {
        for (int j = 0; j < elements; j++) {
            HAADF(i) += (Z(j) / zMean) * std::pow(Q(j, i), gamma);
        }
    }

    return HAADF;
}

Eigen::MatrixXd computeResidualGradient(const Eigen::Ref<const Eigen::VectorXd>& R, const Eigen::Ref<const Eigen::VectorXi>& Z) {
    int pixels = R.size();
    int elements = Z.size();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(elements, pixels);
    double zMean = Z.mean();

    #pragma omp parallel for
    for (int i = 0; i < pixels; i++) {
        for (int j = 0; j < elements; j++) {
            grad(j, i) = (Z(j) / zMean) * R(i);
        }
    }

    return grad;
}

double forwardModelCost(const Eigen::Ref<const Eigen::VectorXd> HAADFInit, const Eigen::Ref<const Eigen::MatrixXd>& Q, const Eigen::Ref<const Eigen::VectorXi>& Z, double gamma) {
    Eigen::VectorXd HAADF = estimateHAADF(Q, Z, gamma);
    Eigen::VectorXd R = HAADF - HAADFInit;

    return 0.5 * R.squaredNorm();
}

double poissonTermCost(const Eigen::Ref<const Eigen::MatrixXd>& QInit, const Eigen::Ref<const Eigen::MatrixXd>& Q, double epsilon) {
    double cost = ((Q.array() + epsilon).log() * QInit.array() - Q.array()).sum();

    return cost;
}

void TVObjective2D(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input, Eigen::Ref<Eigen::MatrixXd> Output, const Eigen::Ref<const Eigen::MatrixXd>& Px, const Eigen::Ref<const Eigen::MatrixXd>& Py, double lambda) {
    double v1, v2;

    #pragma omp parallel for shared(Input, Output, Px, Py, lambda) private(v1, v2)
    for (int i = 0; i < xDim; i++) {
        for (int j = 0; j < yDim; j++) {
            if (i == 0) {
                v1 = 0.0;
            } 
            
            else {
                v1 = Px(i - 1, j);
            }

            if (j == 0) {
                v2 = 0.0;
            } 
            
            else {
                v2 = Py(i, j - 1);
            }

            Output(i, j) = Input(i, j) - lambda * (Px(i, j) + Py(i, j) - v1 - v2);
        }
    }
}

void TVGradient2D(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input, Eigen::Ref<Eigen::MatrixXd> Px, Eigen::Ref<Eigen::MatrixXd> Py, double lFactor) {
    double v1, v2;
    
    #pragma omp parallel for shared(Input, Px, Py, lFactor) private(v1, v2)
    for (int i = 0; i < xDim; i++) {
        for (int j = 0; j < yDim; j++) {
            if (i == xDim - 1) {
                v1 = 0.0;
            }
            
            else {
                v1 = Input(i, j) - Input(i + 1, j);
            }

            if (j == yDim - 1) {
                v2 = 0;
            } 
            
            else {
                v2 = Input(i,j) - Input(i, j + 1);
            }
            
            Px(i, j) += lFactor * v1;
            Py(i, j) += lFactor * v2;
        }
    }
}

void TVProject2D(int xDim, int yDim, Eigen::Ref<Eigen::MatrixXd> Px, Eigen::Ref<Eigen::MatrixXd> Py) {
    double v;

    #pragma omp parallel for shared(Px, Py) private(v)
    for (int i = 0; i < xDim; i++) {
        for (int j = 0; j < yDim; j++) {
            v = Px(i, j) * Px(i, j) + Py(i, j) * Py(i, j);
            
            if (v > 1.0) {
                Px(i, j) /= sqrt(v);
                Py(i, j) /= sqrt(v);
            }
        }
    }
}

Eigen::MatrixXd TVFastGradientProjection(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd> &Input, double lambdaReg, int nIter) {
    Eigen::MatrixXd Px = Eigen::MatrixXd::Zero(xDim, yDim);
    Eigen::MatrixXd Py = Eigen::MatrixXd::Zero(xDim, yDim);
    Eigen::MatrixXd Output = Eigen::MatrixXd::Zero(xDim, yDim);

    double lFactor = 1.0 / (8.0 * lambdaReg);

    for (int iter = 0; iter < nIter; iter++) {
        TVObjective2D(xDim, yDim, Input, Output, Px, Py, lambdaReg);
        TVGradient2D(xDim, yDim, Output, Px, Py, lFactor);
        TVProject2D(xDim, yDim, Px, Py);
    }

    Output = (Output.array() < 0).select(0, Output);

    return Output;
}

double TVRegularisationCost(int xDim, int yDim, const Eigen::Ref<const Eigen::MatrixXd>& Input) {
    double epsilon = 1E-8;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(xDim, yDim);
    
    #pragma omp parallel for
    for (int i = 0; i < xDim; i++)
    {
        int ip = (i + 1) % xDim;
        for (int j = 0; j < yDim; j++)
        {
            int jp = (j + 1) % yDim;
            R(i,j) = sqrt(epsilon + (R(i , j) - R(ip, j)) * (R(i, j) - R(ip, j)) + (R(i, j) - R(i, jp)) * (R(i, j) - R(i, jp)));
        }
    }
    
    return R.sum();
}