#include "../include/EDXSDataset.h"

EDXSDataset::EDXSDataset() {
    _Gcols = 0;
    _beamEnergy = 0;
    _problemType = EDXSModelConstants::problemType::BREMSSTRAHLUNG;
    _absorptionModelType = EDXSModelConstants::absorptionModelType::INTERNAL;
    _azimuthAngle = 0;
    _elevationAngle = 0;
    _tiltStage = 0;
    _thickness = 0;
    _density = 0;
    _widthSlope = 0;
    _widthIntercept = 0;
    _energyAxisSize = 0;
    _energyAxisScale = 0;
    _energyAxisOffset = 0;
    _detectorEfficiency = "";
    _xrayDB = "";
    _massAbsorptionCoefficientsFilePath = "";
    _decompositionResultsFilePath = "";
    _absorptionMatrixFilePath = "";
    _thicknessMapFilePath = "";
    _periodicTableInfoFilePath = "";
    _elements = {};
    _splitLinesElements = {};
    _energyThresholds = {};
    _modelElements = {};
    _quantificationElements = {};
    _absorptionElements = {};
    _absorptionElementsConcentrations = Eigen::VectorXd::Zero(0);
    _energyScale = Eigen::VectorXd::Zero(0);
    _X = Eigen::MatrixXd::Zero(0, 0);
    _norms = Eigen::VectorXd::Zero(0);
    _detectorEfficiencyMatrix = Eigen::MatrixXd::Zero(0, 0);
    _G = Eigen::MatrixXd::Zero(0, 0);
    _T = Eigen::VectorXd::Zero(0);
    _A = Eigen::MatrixXd::Zero(0, 0);
    _xrayDBFile = nlohmann::json();
    _massAbsorptionCoefficientsDBFile = nlohmann::json();
    _periodicTableInfoDBFile = nlohmann::json();
};


EDXSDataset::EDXSDataset(
        int beamEnergy,
        EDXSModelConstants::problemType problemType,
        EDXSModelConstants::absorptionModelType absorptionModelType,
        double azimuthAngle,
        double elevationAngle,
        double tiltStage,
        double thickness,
        double density,
        double widthSlope,
        double widthIntercept,
        double energyAxisSize,
        double energyAxisScale,
        double energyAxisOffset,
        std::string detectorEfficiency,
        std::string xrayDB,
        std::string massAbsorptionCoefficientsFilePath,
        std::string decompositionResultsFilePath,
        std::string absorptionMatrixFilePath,
        std::string thicknessMapFilePath,
        std::string periodicTableInfoFilePath,
        std::vector<std::string> elements,
        std::vector<std::string> splitLinesElements,
        std::vector<double> energyThresholds,
        std::vector<std::string> quantificationElements,
        std::vector<std::string> absorptionElements,
        Eigen::VectorXd absorptionElementsConcentrations
        )
    {
    
    _beamEnergy = beamEnergy;
    _problemType = problemType;
    _absorptionModelType = absorptionModelType;
    _azimuthAngle = azimuthAngle;
    _elevationAngle = elevationAngle;
    _tiltStage = tiltStage;
    _thickness = thickness * 1E-7;
    _density = density;
    _widthSlope = widthSlope;
    _widthIntercept = widthIntercept;
    _energyAxisSize = energyAxisSize;
    _energyAxisScale = energyAxisScale;
    _energyAxisOffset = energyAxisOffset;
    _detectorEfficiency = detectorEfficiency;
    _xrayDB = xrayDB;
    _massAbsorptionCoefficientsFilePath = massAbsorptionCoefficientsFilePath;
    _decompositionResultsFilePath = decompositionResultsFilePath;
    _absorptionMatrixFilePath = absorptionMatrixFilePath;
    _thicknessMapFilePath = thicknessMapFilePath;
    _periodicTableInfoFilePath = periodicTableInfoFilePath;
    _elements = elements;
    _splitLinesElements = splitLinesElements;
    _energyThresholds = energyThresholds;
    _quantificationElements = quantificationElements;
    _absorptionElements = absorptionElements;
    _absorptionElementsConcentrations = absorptionElementsConcentrations;

    _energyScale = buildEnergyScale(_energyAxisSize, _energyAxisScale, _energyAxisOffset);
    _detectorEfficiencyMatrix = readDetectorEfficiency(_detectorEfficiency);
    _Gcols = countGColumns(_elements, _splitLinesElements);

    if (_problemType != EDXSModelConstants::problemType::IDENTITY && _problemType != EDXSModelConstants::problemType::BREMSSTRAHLUNG && _problemType != EDXSModelConstants::problemType::NO_BREMSSTRAHLUNG) {
        throw std::invalid_argument("EDXS Model Error : Invalid problem type.");
    }

    if (_problemType == EDXSModelConstants::problemType::BREMSSTRAHLUNG) {
        _Gcols += 2;
    }
    
    if (_problemType == EDXSModelConstants::problemType::IDENTITY) {
        _Gcols = _energyAxisSize;
        _G = Eigen::MatrixXd::Identity(_Gcols, _Gcols);
        _norms = Eigen::VectorXd::Constant(_Gcols, 1.0);
    }

    else {
        _G = Eigen::MatrixXd::Zero(_energyAxisSize, _Gcols);
        _norms = Eigen::VectorXd::Zero(_Gcols);
    }

    std::ifstream xrayDBFile(_xrayDB);
    
    if (!xrayDBFile.is_open()) {
        throw std::invalid_argument("EDXS Model Error : X-ray database file not found.");
    }

    xrayDBFile >> _xrayDBFile;
    xrayDBFile.close();

    _massAbsorptionCoefficientsDBFile = nlohmann::json();

    if (_absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL || _absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
        std::ifstream massAbsorptionCoefficientsFile(_massAbsorptionCoefficientsFilePath);

        if (!massAbsorptionCoefficientsFile.is_open()) {
            throw std::invalid_argument("EDXS Model Error : Mass absorption coefficient file not found.");
        }

        massAbsorptionCoefficientsFile >> _massAbsorptionCoefficientsDBFile;
        massAbsorptionCoefficientsFile.close();
    }

    std::ifstream periodicTableInfoFile(_periodicTableInfoFilePath);

    if (!periodicTableInfoFile.is_open()) {
        throw std::invalid_argument("EDXS Model Error : Periodic table info file not found.");
    }

    periodicTableInfoFile >> _periodicTableInfoDBFile;
    periodicTableInfoFile.close();
};


EDXSDataset::~EDXSDataset() {
};


void EDXSDataset::generateElementalGMatrix() {
    std::unordered_map<std::string, int> splitLinesElementsIndexMap;

    for (size_t i = 0; i < _splitLinesElements.size(); i++) {
        splitLinesElementsIndexMap[_splitLinesElements[i]] = i;
    }
    
    int idx = 0;

    for (size_t i = 0 ; i < _elements.size(); i++) {
        Eigen::MatrixXd xrayLines = readXrayLines(_elements[i], _xrayDBFile);
        Eigen::MatrixXd peaks = Eigen::MatrixXd::Zero(_energyAxisSize, 2);

        auto it = splitLinesElementsIndexMap.find(_elements[i]);

        if (it != splitLinesElementsIndexMap.end()) {
            Eigen::VectorXd peaksLow = Eigen::VectorXd::Zero(_energyAxisSize);
            Eigen::VectorXd peaksHigh = Eigen::VectorXd::Zero(_energyAxisSize);

            int splitIndex = it->second;

            for (int j = 0; j < xrayLines.rows(); j++) {
                double energy = xrayLines(j, 0);
                double cs = xrayLines(j, 1);

                if (energy > _energyScale(0) && energy < _energyScale(_energyScale.size() - 1)) {
                    Eigen::VectorXd gaussianVector = Eigen::VectorXd::Zero(_energyAxisSize);

                    double width = (_widthSlope * energy + _widthIntercept) / 2.3548;
                    gaussian(_energyScale, energy, width, gaussianVector);

                    int detectorEfficiencyIndex = findClosestValueIndexInVector(_detectorEfficiencyMatrix.col(0), energy);
                    double detectorEfficiencyValue = _detectorEfficiencyMatrix(detectorEfficiencyIndex, 1);
                    double absorptionCorrectionFactor = 1.0;

                    if (_absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
                        absorptionCorrectionFactor = computeSingleAbsorptionCorrection(energy, _elements[i], _thickness, _elevationAngle, &_density, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
                    }

                    if (energy < _energyThresholds[splitIndex]) {
                        peaksLow += cs * gaussianVector * detectorEfficiencyValue * absorptionCorrectionFactor;
                    }

                    else {
                        peaksHigh += cs * gaussianVector * detectorEfficiencyValue * absorptionCorrectionFactor;
                    }
                }
            }

            if (peaksLow.maxCoeff() > 0.0 && peaksHigh.maxCoeff() > 0.0) {
                _G.col(idx) = peaksLow;
                _modelElements.push_back(_elements[i] + "_low");
                idx++;

                _G.col(idx) = peaksHigh;
                _modelElements.push_back(_elements[i] + "_high");
                idx++;
            }

            else {
                std::cout <<"Element " << _elements[i] <<" : No X-ray peaks are present in the energy range of the dataset." << "\n";
            }
        }

        else {
            Eigen::VectorXd peaks = Eigen::VectorXd::Zero(_energyAxisSize);

            for (int j = 0; j < xrayLines.rows(); j++) {
                double energy = xrayLines(j, 0);
                double cs = xrayLines(j, 1);

                if (energy > _energyScale(0) && energy < _energyScale(_energyScale.size() - 1)) {
                    Eigen::VectorXd gaussianVector = Eigen::VectorXd::Zero(_energyAxisSize);

                    double width = (_widthSlope * energy + _widthIntercept) / 2.3548;
                    gaussian(_energyScale, energy, width, gaussianVector);

                    int detectorEfficiencyIndex = findClosestValueIndexInVector(_detectorEfficiencyMatrix.col(0), energy);
                    double detectorEfficiencyValue = _detectorEfficiencyMatrix(detectorEfficiencyIndex, 1);
                    double absorptionCorrectionFactor = 1.0;

                    if (_absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
                        absorptionCorrectionFactor = computeSingleAbsorptionCorrection(energy, _elements[i], _thickness, _elevationAngle, &_density, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
                    }

                    peaks += cs * gaussianVector * detectorEfficiencyValue * absorptionCorrectionFactor;
                }
            }

            if (peaks.maxCoeff() > 0.0) {
                _G.col(idx) = peaks;
                _modelElements.push_back(_elements[i]);
                idx++;
            }

            else {
                std::cout <<"Element " << _elements[i] <<" : No X-ray peaks are present in the energy range of the dataset." << "\n";
            }
        }
    }
}

void EDXSDataset::generateBremsstrahlungGMatrix() {
    Eigen::VectorXd detectorEfficiencyVector = Eigen::VectorXd::Zero(_energyAxisSize);

    for (int i = 0; i < _energyScale.size(); i++) {
        int detectorEfficiencyIndex = findClosestValueIndexInVector(_detectorEfficiencyMatrix.col(0), _energyScale(i));
        detectorEfficiencyVector(i) = _detectorEfficiencyMatrix(detectorEfficiencyIndex, 1);
    }

    Eigen::VectorXd A = Eigen::VectorXd::Constant(_energyAxisSize, 1.0);

    if (_absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
        if (!_absorptionElements.empty()) {
            if (!_absorptionElementsConcentrations.isZero()) {
                A = computeAbsorptionCorrection(_energyScale, _absorptionElements, _absorptionElementsConcentrations, _thickness, _elevationAngle, &_density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
            }

            else {
                Eigen::VectorXd concentrations = Eigen::VectorXd::Constant(_absorptionElements.size(), 1.0 / _absorptionElements.size());
                A = computeAbsorptionCorrection(_energyScale, _absorptionElements, concentrations, _thickness, _elevationAngle, &_density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
            }
        }

        else {
            Eigen::VectorXd concentrations = Eigen::VectorXd::Constant(_elements.size(), 1.0 / _elements.size());
            A = computeAbsorptionCorrection(_energyScale, _elements, concentrations, _thickness, _elevationAngle, &_density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
        }
    }

    Eigen::VectorXd B0 = A.array() * detectorEfficiencyVector.array() * lifshinBremsstrahlungB0(_energyScale, 1.0, _beamEnergy).array();
    Eigen::VectorXd B1 = A.array() * detectorEfficiencyVector.array() * lifshinBremsstrahlungB1(_energyScale, 1.0, _beamEnergy).array();

    if (B0.maxCoeff() > 0.0 && B1.maxCoeff() > 0.0) {
        _G.col(_Gcols - 2) = B0;
        _G.col(_Gcols - 1) = B1;
    }

    else {
        std::cerr <<"Check Bremsstrahlung parameters."<<"\n";
    }
}

void EDXSDataset::generateGMatrix() {
    generateElementalGMatrix();

    if (_problemType == EDXSModelConstants::problemType::BREMSSTRAHLUNG) {
        generateBremsstrahlungGMatrix();
    }

    _norms = _G.colwise().squaredNorm().cwiseSqrt();

    if (_problemType == EDXSModelConstants::problemType::BREMSSTRAHLUNG) {
        double normMean = _norms(Eigen::seq(0, _Gcols - 3)).mean();
        _norms(Eigen::seq(0, _Gcols - 3)) = Eigen::VectorXd::Constant(_Gcols - 2, normMean);
    }

    else {
        double normMean = _norms.mean();
        _norms = Eigen::VectorXd::Constant(_G.cols(), normMean);
    }

    for (int i = 0; i < _G.rows(); i++) {
        _G.row(i).array() /= _norms.array().transpose();
    }
}

std::vector<int> EDXSDataset::NMFSimplexIndices() {
    std::regex pattern(R"(^[A-Za-z]+_low$)");
    std::vector<int> simplexIndices;

    for (size_t i = 0; i < _modelElements.size(); i++) {
        if (std::regex_match(_modelElements[i], pattern)) {
            continue;
        }

        else {
            simplexIndices.push_back(i);
        }
    }

    return simplexIndices;
}

std::vector<std::string> EDXSDataset::NMFSimplexElements() {
    std::regex pattern_low(R"(^[A-Za-z0-9]+_low$)");
    std::regex pattern_high(R"(^([A-Za-z0-9]+)_high$)");
    std::vector<std::string> simplexElements;

    for (size_t i = 0; i < _modelElements.size(); i++) {
        if (std::regex_match(_modelElements[i], pattern_low)) {
            continue;
        }

        else if (std::regex_match(_modelElements[i], pattern_high)) {
            std::smatch match;
            std::regex_search(_modelElements[i], match, pattern_high);
            simplexElements.push_back(match.str(1));
        }

        else {
            simplexElements.push_back(_modelElements[i]);
        }
    }

    return simplexElements;
}

std::vector<int> EDXSDataset::selectedElementsIndices(const std::vector<std::string>& selectedElements) {
    std::regex pattern_low(R"(^[A-Za-z]+_low$)");
    std::regex pattern_high(R"(^([A-Za-z0-9]+)_high$)");
    std::vector<int> selectedElementsIndices;

    for (size_t i = 0; i < selectedElements.size(); i++) {
        for (size_t j = 0; j < _modelElements.size(); j++) {
            std::string element;
            
            if (std::regex_match(_modelElements[j], pattern_low)) {
                continue;
            }

            else if (std::regex_match(_modelElements[j], pattern_high)) {
                std::smatch match;
                std::regex_search(_modelElements[j], match, pattern_high);
                element = match.str(1);
            }

            else {
                element = _modelElements[j];
            }

            if (element == selectedElements[i]) {
                selectedElementsIndices.push_back(j);
            }
        }
    }

    return selectedElementsIndices;
}

void EDXSDataset::updateGBremsstrahlung(const Eigen::Ref<const Eigen::MatrixXd>& W) {
    std::vector<int> simplexIndices;
    std::vector<std::string> elements;

    if (!_absorptionElements.empty()) {
        simplexIndices = selectedElementsIndices(_absorptionElements);
        elements = _absorptionElements;
    }

    else {
        simplexIndices = NMFSimplexIndices();
        elements = NMFSimplexElements();
    }

    Eigen::VectorXd componentMean = Eigen::VectorXd::Zero(simplexIndices.size());

    for (size_t i = 0; i < simplexIndices.size(); i++) {
        componentMean(i) = W.row(simplexIndices[i]).mean();
    }

    double sum = componentMean.sum();
    Eigen::VectorXd normedComponents = (componentMean.array() / sum).array() * 100.0;

    Eigen::VectorXd detectorEfficiencyVector = Eigen::VectorXd::Zero(_energyAxisSize);

    for (int i = 0; i < _energyScale.size(); i++) {
        int detectorEfficiencyIndex = findClosestValueIndexInVector(_detectorEfficiencyMatrix.col(0), _energyScale(i));
        detectorEfficiencyVector(i) = _detectorEfficiencyMatrix(detectorEfficiencyIndex, 1);
    }

    Eigen::VectorXd A = computeAbsorptionCorrection(_energyScale, elements, normedComponents, _thickness, _elevationAngle, &_density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);

    Eigen::VectorXd B0 = A.array() * detectorEfficiencyVector.array() * lifshinBremsstrahlungB0(_energyScale, 1.0, _beamEnergy).array();
    Eigen::VectorXd B1 = A.array() * detectorEfficiencyVector.array() * lifshinBremsstrahlungB1(_energyScale, 1.0, _beamEnergy).array();

    _G.col(_G.cols() - 2) = B0.array() / _norms(_G.cols() - 2);
    _G.col(_G.cols() - 1) = B1.array() / _norms(_G.cols() - 1);
}

Eigen::VectorXd EDXSDataset::readThicknessMap() {
    std::ifstream thicknessMapFile(_thicknessMapFilePath);

    int pixels;

    thicknessMapFile >> pixels;

    if (!thicknessMapFile.is_open()) {
        throw std::invalid_argument("EDXS Model Error : Thickness map file not found.");
    }

    Eigen::VectorXd thicknessMap = Eigen::VectorXd::Zero(pixels);

    for (int i = 0; i < pixels; i++) {
        thicknessMapFile >> thicknessMap(i);
    }

    thicknessMapFile.close();

    return thicknessMap;
}

Eigen::MatrixXd EDXSDataset::generateQuantificationMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H) {
    Eigen::MatrixXd WH = W * H;
    std::vector<int> simplexIndices = NMFSimplexIndices();
    Eigen::MatrixXd Q = WH(simplexIndices, Eigen::placeholders::all);
    
    #pragma omp parallel for
    for (int j = 0; j < Q.cols(); j++) {
        double colWiseSum = Q.col(j).sum();
        Q.col(j).array() /= colWiseSum;
    }

    return Q.array() * 100.0;
}

Eigen::MatrixXd EDXSDataset::generateQuantificationMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, const std::vector<std::string>& selectedElements) {
    Eigen::MatrixXd WH = W * H;
    std::vector<int> selectedElementsIdx = selectedElementsIndices(selectedElements);
    Eigen::MatrixXd Q = WH(selectedElementsIdx, Eigen::placeholders::all);
    
    #pragma omp parallel for
    for (int j = 0; j < Q.cols(); j++) {
        double colWiseSum = Q.col(j).sum();
        Q.col(j).array() /= colWiseSum;
    }

    return Q.array() * 100.0;
}

Eigen::VectorXd EDXSDataset::computeDensityMap(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H) {
    Eigen::VectorXd D = Eigen::VectorXd::Zero(H.cols());

    Eigen::MatrixXd Q;
    std::vector<std::string> elements;

    if (!_absorptionElements.empty()) {
        elements = _absorptionElements;
        Q = generateQuantificationMatrix(W, H, elements);
    }

    else {
        elements = NMFSimplexElements();
        Q = generateQuantificationMatrix(W, H);
    }

    for (int j = 0; j < Q.cols(); j++) {
        Eigen::VectorXd weightPercentages = atomicToWeightPercent(Q.col(j), elements, _periodicTableInfoDBFile);
        D(j) = densityOfMixture(weightPercentages, elements, _periodicTableInfoDBFile, EDXSModelConstants::meanType::HARMONIC);
    }

    return D;
}

Eigen::VectorXd EDXSDataset::computeDensityMap(const Eigen::Ref<const Eigen::MatrixXd>& Q, const std::vector<std::string>& elements) {
    Eigen::VectorXd D = Eigen::VectorXd::Zero(Q.cols());

    if (elements.size() != Q.rows()) {
        throw std::invalid_argument("EDXS Model Error : Number of elements and rows in quantification data matrix do not match.");
    }

    for (int j = 0; j < Q.cols(); j++) {
        Eigen::VectorXd weightPercentages = atomicToWeightPercent(Q.col(j), elements, _periodicTableInfoDBFile);
        D(j) = densityOfMixture(weightPercentages, elements, _periodicTableInfoDBFile, EDXSModelConstants::meanType::HARMONIC);
    }

    return D;
}

Eigen::MatrixXd EDXSDataset::generateAbsorptionCorrectionMatrix(const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(_energyAxisSize, H.cols());

    Eigen::MatrixXd Q;
    std::vector<std::string> elements;

    if (!_absorptionElements.empty()) {
        elements = _absorptionElements;
        Q = generateQuantificationMatrix(W, H, elements);
    }

    else {
        elements = NMFSimplexElements();
        Q = generateQuantificationMatrix(W, H);
    }

    Eigen::VectorXd thicknessMap = readThicknessMap().array() * 1E-7;

    #pragma omp parallel for
    for (int j = 0; j < Q.cols(); j++) {
        double density = 0.0;
        Eigen::VectorXd absorptionCorrection = computeAbsorptionCorrection(_energyScale, elements, Q.col(j), thicknessMap(j), _elevationAngle, &density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
        A.col(j) = absorptionCorrection;
    }

    return A;
}

Eigen::MatrixXd EDXSDataset::generateAbsorptionCorrectionMatrix(const Eigen::Ref<const Eigen::MatrixXd>& Q, const std::vector<std::string>& elements){
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(_energyAxisSize, Q.cols());
    Eigen::VectorXd thicknessMap = Eigen::VectorXd::Zero(Q.cols());

    if (!_thicknessMapFilePath.empty()) {
        Eigen::VectorXd thicknessMap = readThicknessMap().array() * 1E-7;
    }

    else {
        thicknessMap = Eigen::VectorXd::Constant(Q.cols(), _thickness);
    }

    #pragma omp parallel for
    for (int j = 0; j < Q.cols(); j++) {
        double density = 0.0;
        Eigen::VectorXd absorptionCorrection = computeAbsorptionCorrection(_energyScale, elements, Q.col(j), thicknessMap(j), _elevationAngle, &density, true, _periodicTableInfoDBFile, _massAbsorptionCoefficientsDBFile);
        A.col(j) = absorptionCorrection;
    }

    return A;
}

void EDXSDataset::applyAbsorptionCorrection(Eigen::Ref<Eigen::MatrixXd> X) {
    if (!_absorptionMatrixFilePath.empty()) {
        Eigen::MatrixXd A = readMatrixFromFile(_absorptionMatrixFilePath);
        X = X.array() / A.array();
    }
}

