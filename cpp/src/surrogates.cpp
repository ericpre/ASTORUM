#include "../include/surrogates.h"

double smoothL2Surrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, double sigmaL, double lambdaL) {
    Eigen::MatrixXd Holdsq = Hold.cwiseProduct(Hold);
    double temp0 = Holdsq.sum();
    double temp1;
    double temp2;

    if (H.isZero()) {
        temp1 = temp0;
        temp2 = 0.0;
    }

    else {
        Eigen::MatrixXd HoldH = Hold.cwiseProduct(H);
        temp1 = HoldH.sum();
        Eigen::MatrixXd diff = Hold - H;
        Eigen::MatrixXd sq = diff.cwiseProduct(diff);
        temp2 = sq.sum();
    }

    return (lambdaL / 2.0 * (2.0 * temp1 - temp0 + sigmaL * temp2));
}

double smoothDGKLSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, double sigmaL, double lambdaL) {
    Eigen::MatrixXd Holdsq = Hold.cwiseProduct(Hold);
    double temp0 = Holdsq.sum();
    double temp1;
    double temp2;

    if (H.isZero()) {
        temp1 = temp0;
        temp2 = 0.0;
    }

    else {
        Eigen::MatrixXd HoldH = Hold.cwiseProduct(H);
        temp1 = HoldH.sum();
        
        Eigen::VectorXd maxHrows = H.rowwise().maxCoeff();
        Eigen::MatrixXd ratio = Hold.cwiseQuotient(H);
        Eigen::MatrixXd logRatio;
        logRatio.array() = ratio.array().log();
        Eigen::MatrixXd prod = logRatio.cwiseProduct(Hold);
        Eigen::MatrixXd tempDiff;
        tempDiff.array() = prod.array() - Hold.array();
        Eigen::MatrixXd dgkl;
        dgkl.array() = tempDiff.array() + H.array();
        Eigen::VectorXd sumDgklRows = dgkl.rowwise().sum();
        Eigen::VectorXd rowProd = maxHrows.cwiseProduct(sumDgklRows);
        temp2 = rowProd.sum();
    }

    return (lambdaL / 2.0 * (2.0 * temp1 - temp0 + sigmaL * temp2));
}

double diffSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& Hold, const Eigen::Ref<const Eigen::MatrixXd>& H, double sigmaL, double lambdaL, SmoothNMFConstants::algorithm algorithm) {
    Eigen::MatrixXd HT = H.transpose();
    double b0;
    double b1;

    b0 = traceXTX(HT, false) * lambdaL / 2.0;

    if (algorithm == SmoothNMFConstants::algorithm::LOG_SURROGATE || algorithm == SmoothNMFConstants::algorithm::BMD) 
        b1 = smoothDGKLSurrogate(Hold, H, sigmaL, lambdaL);
    
    else if (algorithm == SmoothNMFConstants::algorithm::L2_SURROGATE)
        b1 = smoothL2Surrogate(Hold, H, sigmaL, lambdaL);
    
    else
        throw std::invalid_argument("Surrogates Error : Invalid surrogate algorithm.");

    return (b1 - b0);
}

double quadraticSurrogate(const Eigen::Ref<const Eigen::MatrixXd>& A, const Eigen::Ref<const Eigen::MatrixXd>& Aold, const Eigen::Ref<const Eigen::MatrixXd>& grad, double lossOld, double gammaVal) {
    Eigen::MatrixXd diff = A - Aold;
    Eigen::MatrixXd tempSq = diff.cwiseProduct(diff);
    Eigen::MatrixXd tempProd = diff.cwiseProduct(grad);

    return (lossOld + tempProd.sum() + gammaVal * tempSq.sum());
}