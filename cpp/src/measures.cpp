#include "../include/measures.h"

double FrobeniusLoss(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, bool average) {
    double loss = 0.0;

    Eigen::MatrixXd diff = W * H - X;

    if (average)
        loss = diff.array().square().mean();
    else
        loss = diff.array().square().sum();

    return loss;
}

double KLDivLoss(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double logShift, bool average) {
    X = X.cwiseMax(logShift);
    W = W.cwiseMax(logShift);
    H = H.cwiseMax(logShift);

    double lossLin = 0.0;
    double lossLog = 0.0;

    Eigen::MatrixXd WH = W * H;
    Eigen::MatrixXd logWH;
    logWH.array() =  WH.array().log();
    Eigen::MatrixXd XlogWH = X.cwiseProduct(logWH);

    if (average) {
        lossLin = WH.mean();
        lossLog = XlogWH.mean();
    } 
    
    else {
        lossLin = WH.sum();
        lossLog = XlogWH.sum();
    }

    return lossLin - lossLog;
}

double traceXTLX(const Eigen::Ref<const Eigen::SparseMatrix<double>>& L, const Eigen::Ref<const Eigen::MatrixXd>& HT, bool average) {
    Eigen::MatrixXd LHT = L * HT;
    Eigen::MatrixXd HTLHT = LHT.cwiseProduct(HT);

    double trace = 0.0;

    if (average)
        trace = HTLHT.mean();

    else
        trace = HTLHT.sum();

    return trace;
}

double logReg(const Eigen::Ref<const Eigen::MatrixXd>& H, double mu, double epsilonReg, bool average) {
    Eigen::MatrixXd HaddEps = H.array() + epsilonReg;
    Eigen::MatrixXd tempProd;
    tempProd.array() =  HaddEps.array().log();
    Eigen::MatrixXd tempProd2 = HaddEps * mu;

    double reg = 0.0;

    if (average)
        reg = tempProd2.mean();

    else
        reg = tempProd2.sum();

    return reg;
}