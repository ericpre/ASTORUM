#ifndef HAADFDATASET_H
#define HAADFDATASET_H

#include "fusionutils.h"

class HAADFDataset {
    public:
        int _xDim;
        int _yDim;
        Eigen::VectorXd _XInit;
        Eigen::VectorXd _X;
        Eigen::VectorXi _Z;
        Eigen::MatrixXd _QInit;
        Eigen::MatrixXd _Q;
        std::string _periodicTableInfoFilePath;
        std::vector<std::string> _elements;
        nlohmann::json _periodicTableInfoDBFile;
    
    public:
        HAADFDataset();

        HAADFDataset(
            int xDim,
            int yDim,
            const Eigen::Ref<const Eigen::VectorXd>& XInit
        );

        ~HAADFDataset();

        void loadEDXSQuantificationData(const Eigen::Ref<const Eigen::MatrixXd>& QInit, std::vector<std::string> elements, std::string periodicTableInfoFilePath);

        void runQuantificationDataOptimisationRoutine(double gamma, double lambdaHAADF, double lambdaChem, double lambdaTV, double epsilon, int nIter, int nIterTV, Eigen::Ref<Eigen::VectorXd> costHAADF, Eigen::Ref<Eigen::VectorXd> costChem, Eigen::Ref<Eigen::VectorXd> costTV, bool regularise);
};

#endif