#ifndef EELSDATASET_H
#define EELSDATASET_H

#include "modelutils.h"

class EELSDataset {
    public:
        Eigen::MatrixXd _X;
        Eigen::VectorXd _energyAxis;
        int _channels;
        int _pixels;
        double _energyAxisScale;
        double _energyAxisOffset;
        double _electronEnergy;
        Eigen::VectorXd _densityMap;
        double _alpha;
        double _beta;
        double _zeroLossPeakThreshold;
        Eigen::VectorXd _T;

    public:
        EELSDataset();

        EELSDataset(
            const Eigen::Ref<const Eigen::MatrixXd>& X,
            const Eigen::Ref<const Eigen::VectorXd>& energyAxis,
            double energyAxisScale,
            double energyAxisOffset,
            double electronEnergy,
            const Eigen::Ref<const Eigen::VectorXd>& densityMap,
            double alpha,
            double beta,
            double zeroLossPeakThreshold
            );

        ~EELSDataset();

        void computeThicknessMap();
};

#endif