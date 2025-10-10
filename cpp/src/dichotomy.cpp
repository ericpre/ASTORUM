#include "../include/dichotomy.h"

void functionSimplex(const Eigen::Ref<const Eigen::VectorXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& num, const Eigen::Ref<const Eigen::MatrixXd>& denum, double logShift, Eigen::Ref<Eigen::VectorXd> R) {
    Eigen::MatrixXd temp0 = denum.array() + X.transpose().replicate(denum.rows(), 1).array();
    Eigen::MatrixXd temp1 = num.cwiseQuotient(temp0).cwiseMax(logShift);

    R = temp1.colwise().sum();
    R.array() -= 1.0;
}

void dichotomySimplex(const Eigen::Ref<const Eigen::MatrixXd>& A, const Eigen::Ref<const Eigen::MatrixXd>& B, double logShift, double tol, int maxIter, Eigen::Ref<Eigen::VectorXd> R) {  
    if ((A.array() == 0).any())
        throw std::invalid_argument("Dichotomy Error : The matrix A must be non-negative.");
    
    if ((B.array() == 0).any())
        throw std::invalid_argument("Dichotomy Error : The matrix B must be non-negative.");
    
    if ((A.colwise().sum().array() <= 0).any())
        throw std::invalid_argument("Dichotomy Error : The column sum of A must be strictly positive.");

    if (logShift > 0.0) {
        if (A.rows() * logShift >= 1.0)
            throw std::invalid_argument("Dichotomy Error : No solution exists.");

        double scale = 1.0 / logShift;

        Eigen::MatrixXd Bmax = B.array() * scale;
    }

    Eigen::VectorXd tempA(A.cols());
    Eigen::VectorXd tempB(A.cols());

    double tempMax;

    for (int i = 0; i < A.cols(); i++) {
        tempMax = -1E6;
        for (int j = 0; j < A.rows(); j++) {
            if (A(j, i) > 0.0)
                tempMax = std::max(tempMax, (A(j, i) / 2 - B(j, i)));
        }

        tempA(i) = tempMax;
    }

    tempB.array() = 2 * A.rows() * A.colwise().maxCoeff().array() - B.colwise().minCoeff().array();

    Eigen::VectorXd funcA(A.cols());
    Eigen::VectorXd funcB(A.cols());

    functionSimplex(tempA, A, B, logShift, funcA);
    functionSimplex(tempB, A, B, logShift, funcB);

    if ((funcA.array().isNaN().any() || funcB.array().isNaN().any()))
        throw std::invalid_argument("Dichotomy Error : The function is not defined for some values.");

    if ((funcA.array() <= 0).any())
        throw std::invalid_argument("Dichotomy Error : The maximum interval function must be strictly positive.");

    if ((funcB.array() >= 0).any())
        throw std::invalid_argument("Dichotomy Error : The minimum interval function must be strictly negative.");

    int iter = 0;
    Eigen::VectorXd funcNew(A.cols());

    R.array() = tempA.array() + tempB.array();
    R.array() *= 0.5;

    functionSimplex(R, A, B, logShift, funcNew);

    Eigen::VectorXd absNew;
    absNew.array() = funcNew.array().abs();
    double maxNew = absNew.maxCoeff();

    while (maxNew > tol) {
        iter++;
        functionSimplex(tempA, A, B, logShift, funcA);

        for (int i = 0; i < A.cols(); i++) {
            if (funcA(i) * funcNew(i) > 0.0)
                tempA(i) = R(i);
            else
                tempB(i) = R(i);
        }
        
        R.array() = tempA.array() + tempB.array();
        R.array() *= 0.5;

        functionSimplex(R, A, B, logShift, funcNew);
        absNew.array() = funcNew.array().abs();
        maxNew = absNew.maxCoeff();

        if (iter >= maxIter) {
            std::cout<<"Dichotomy stopped for maximum number of iterations with an error of"<<" "<<maxNew<<".\n";
            break;
        }
    }
}

void functionSimplexProjectedGradient(const Eigen::Ref<const Eigen::VectorXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& A, double logShift, Eigen::Ref<Eigen::VectorXd> R) {
    Eigen::MatrixXd temp0 = A.array() + X.transpose().replicate(A.rows(), 1).array();
    temp0 = temp0.cwiseMax(logShift).eval();
    
    R = temp0.colwise().sum();
    R.array() -= 1.0;
}

void dichotomySimplexProjectedGradient(const Eigen::Ref<const Eigen::MatrixXd>& A, double logShift, double tol, int maxIter, Eigen::Ref<Eigen::VectorXd> R) {
    if (logShift > 0) {
        if (A.rows() * logShift >= 1.0)
            throw std::invalid_argument("Dichotomy Error : No solution exists.");
    }

    Eigen::VectorXd nuMin(A.cols());
    Eigen::VectorXd nuMax(A.cols());
    nuMin = A.colwise().maxCoeff();
    nuMax = A.colwise().minCoeff();
    nuMin.array() *= -1.0;
    nuMax.array() = -1.0 * nuMax.array() + 1.0 / A.rows();

    Eigen::VectorXd funcA(A.cols());
    Eigen::VectorXd funcB(A.cols());

    functionSimplexProjectedGradient(nuMax, A, logShift, funcA);
    functionSimplexProjectedGradient(nuMin, A, logShift, funcB);

    if ((funcA.array().isNaN().any() || funcB.array().isNaN().any()))
        throw std::invalid_argument("Dichotomy Error : The function is not defined for some values.");

    if ((funcA.array() <= 0).any())
        throw std::invalid_argument("Dichotomy Error : The maximum interval function must be strictly positive.");

    if ((funcB.array() >= 0).any())
        throw std::invalid_argument("Dichotomy Error : The minimum interval function must be strictly negative.");

    int iter = 0;
    Eigen::VectorXd funcNew(A.cols());

    R.array() = nuMax.array() + nuMin.array();
    R.array() *= 0.5;

    functionSimplexProjectedGradient(R, A, logShift, funcNew);

    Eigen::VectorXd absNew;
    absNew.array() = funcNew.array().abs();
    double maxNew = absNew.maxCoeff();

    while (maxNew > tol) {
        iter++;

        functionSimplexProjectedGradient(nuMax, A, logShift, funcA);

        for (int i = 0; i < A.cols(); i++) {
            if (funcA(i) * funcNew(i) > 0.0)
                nuMax(i) = R(i);
            else
                nuMin(i) = R(i);
        }

        R.array() = nuMax.array() + nuMin.array();
        R.array() *= 0.5;

        functionSimplexProjectedGradient(R, A, logShift, funcNew);

        absNew.array() = funcNew.array().abs();
        maxNew = absNew.maxCoeff();

        if (iter >= maxIter) {
            std::cout<<"Dichotomy stopped for maximum number of iterations with an error of"<<" "<<maxNew<<".\n";
            break;
        }
    }
}
