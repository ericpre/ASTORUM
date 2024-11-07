#include "../include/HAADFDataset.h"

HAADFDataset::HAADFDataset() {
    _xDim = 0;
    _yDim = 0;
    _XInit = Eigen::VectorXd::Zero(0);
    _X = Eigen::VectorXd::Zero(0);
    _Z = Eigen::VectorXi::Zero(0);
    _QInit = Eigen::MatrixXd::Zero(0, 0);
    _Q = Eigen::MatrixXd::Zero(0, 0);
    _periodicTableInfoFilePath = "";
    _elements = {};
    _periodicTableInfoDBFile = nlohmann::json();
};


HAADFDataset::HAADFDataset(int xDim, int yDim, const Eigen::Ref<const Eigen::VectorXd>& XInit)  {
    _xDim = xDim;
    _yDim = yDim;
    _XInit = XInit;
    _X = Eigen::VectorXd::Zero(_xDim * _yDim);
    _Z = Eigen::VectorXi::Zero(0);
    _QInit = Eigen::MatrixXd::Zero(0, 0);
    _Q = Eigen::MatrixXd::Zero(0, 0);
    _periodicTableInfoFilePath = "";
    _elements = {};
    _periodicTableInfoDBFile = nlohmann::json();
};


HAADFDataset::~HAADFDataset() {
};


void HAADFDataset::loadEDXSQuantificationData(const Eigen::Ref<const Eigen::MatrixXd>& QInit, std::vector<std::string> elements, std::string periodicTableInfoFilePath) {
    _QInit = QInit;
    _periodicTableInfoFilePath = periodicTableInfoFilePath;
    _elements = elements;
    _Z = Eigen::VectorXi::Zero(_elements.size());
    
    std::ifstream periodicTableInfoFile(_periodicTableInfoFilePath);

    if (!periodicTableInfoFile.is_open()) {
        throw std::invalid_argument("Could not open periodic table information file.");
    }

    periodicTableInfoFile >> _periodicTableInfoDBFile;
    periodicTableInfoFile.close();

    for (int i = 0; i < _elements.size(); i++) {
        if (_periodicTableInfoDBFile["table"].contains(_elements[i])) {
            _Z(i) = _periodicTableInfoDBFile["table"][_elements[i]]["number"];
        }

        else {
            throw std::invalid_argument("Element " + elements[i] + " not found in periodic table information file.");
        }
    }

    if (_QInit.rows() != _elements.size()) {
        throw std::invalid_argument("Number of quantification maps must match the number of specified elements.");
    }
}

void HAADFDataset::runQuantificationDataOptimisationRoutine(double gamma, double lambdaHAADF, double lambdaChem, double lambdaTV, double epsilon, int nIter, int nIterTV, Eigen::Ref<Eigen::VectorXd> costHAADF, Eigen::Ref<Eigen::VectorXd> costChem, Eigen::Ref<Eigen::VectorXd> costTV, bool regularise) {
    Eigen::VectorXd minValues = Eigen::VectorXd::Zero(_elements.size());
    Eigen::VectorXd maxValues = Eigen::VectorXd::Zero(_elements.size());

    for (int i = 0; i < _elements.size(); i++) {
        minValues(i) = _QInit.row(i).minCoeff();
        maxValues(i) = _QInit.row(i).maxCoeff();
        _QInit.row(i).array() -= minValues(i);
        _QInit.row(i).array() /= (maxValues(i) - minValues(i));
    }

    _Q = _QInit;

    double xMin = _XInit.minCoeff();
    double xMax = _XInit.maxCoeff();

    _XInit.array() -= xMin;
    _XInit.array() /= (xMax - xMin);

    if (lambdaHAADF == 0) {
        lambdaHAADF = 1.0 / _elements.size();
    }

    costHAADF = Eigen::VectorXd::Zero(nIter);
    costChem = Eigen::VectorXd::Zero(nIter);
    costTV = Eigen::VectorXd::Zero(nIter);

    for (int iter = 0; iter < nIter; iter++) {
        _X = estimateHAADF(_Q, _Z, gamma);
        Eigen::VectorXd R = _X - _XInit;
        Eigen::MatrixXd grad = computeResidualGradient(R, _Z);

        _Q.array() -= gamma * (_Q.array().pow(gamma - 1)) * lambdaHAADF * grad.array() + lambdaChem * (1.0 - _QInit.array() / (_Q.array() + epsilon));
        _Q = (_Q.array() < 0).select(0, _Q);

        if (regularise) {
            for (int i = 0; i < _elements.size(); i++) {
                Eigen::MatrixXd QMap = _Q.row(i);
                QMap.resize(_xDim, _yDim);

                _Q.row(i) = TVFastGradientProjection(_xDim, _yDim, QMap, lambdaTV, nIterTV).reshaped();

                Eigen::MatrixXd QReg = _Q.row(i);
                QReg.resize(_xDim, _yDim);

                costTV(iter) += TVRegularisationCost(_xDim, _yDim, QReg);
            }
        }

        costHAADF(iter) = forwardModelCost(_XInit, _Q, _Z, gamma);
        costChem(iter) = poissonTermCost(_QInit, _Q, 1E-8);
    }

    for (int i = 0; i < _elements.size(); i++) {
        _Q.row(i).array() *= (maxValues(i) - minValues(i));
        _Q.row(i).array() += minValues(i);

        _QInit.row(i).array() *= (maxValues(i) - minValues(i));
        _QInit.row(i).array() += minValues(i);
    }

    #pragma omp parallel for
    for (int j = 0; j < _Q.cols(); j++) {
        double QColSum = _Q.col(j).sum();
        _Q.col(j) /= QColSum;
    }

    _Q.array() *= 100.0;

    _X.array() *= (xMax - xMin);
    _X.array() += xMin;

    _XInit.array() *= (xMax - xMin);
    _XInit.array() += xMin;
}