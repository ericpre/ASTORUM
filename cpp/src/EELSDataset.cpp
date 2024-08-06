#include "../include/EELSDataset.h"

EELSDataset::EELSDataset() {
    _X = Eigen::MatrixXd::Zero(0, 0);
    _energyAxis = Eigen::VectorXd::Zero(0);
    _channels = 0;
    _pixels = 0;
    _energyAxisScale = 0.0;
    _energyAxisOffset = 0.0;
    _electronEnergy = 0.0;
    _densityMap = Eigen::VectorXd::Zero(0);
    _alpha = 0.0;
    _beta = 0.0;
    _zeroLossPeakThreshold = 0.0;
    _T = Eigen::VectorXd::Zero(0);
};


EELSDataset::EELSDataset(
        const Eigen::Ref<const Eigen::MatrixXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& energyAxis,
        double energyAxisScale,
        double energyAxisOffset,
        double electronEnergy,
        const Eigen::Ref<const Eigen::VectorXd>& densityMap,
        double alpha,
        double beta,
        double zeroLossPeakThreshold
        )
    {

    _X = X;
    _energyAxis = energyAxis;
    _channels = X.rows();
    _pixels = X.cols();
    _energyAxisScale = energyAxisScale;
    _energyAxisOffset = energyAxisOffset;
    _electronEnergy = electronEnergy;
    _densityMap = densityMap;
    _alpha = alpha;
    _beta = beta;
    _zeroLossPeakThreshold = zeroLossPeakThreshold;
    _T = Eigen::VectorXd::Zero(_pixels);
};


EELSDataset::~EELSDataset() {
};


void EELSDataset::computeThicknessMap() {
    #pragma omp parallel for
    for (int i = 0; i < _pixels; i++) {
        _T(i) = estimateThicknessAtPixel(_X.col(i), _energyAxis, _energyAxisScale, _energyAxisOffset, _zeroLossPeakThreshold, _densityMap(i), _electronEnergy, _alpha, _beta);
    }
}