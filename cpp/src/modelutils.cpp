#include "../include/modelutils.h"

void writeMatrixToFile(const Eigen::Ref<const Eigen::MatrixXd>& A, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file : " + filename + " !");
    }

    file << A.rows() << " " << A.cols() << "\n";

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0 ; j < A.cols(); j++)
            file << A(i, j) << " ";
        file << "\n";
    }

    file.close();
}

void writeVectorToFile(const Eigen::Ref<const Eigen::VectorXd>& v, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file : " + filename + " !");
    }

    file << v.size() << "\n";

    for (int i = 0; i < v.size(); i++) {
        file << v(i) << " ";
    }

    file.close();
}

Eigen::MatrixXd readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);

    int rows, cols;

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file : " + filename + " !");
    }

    file >> rows >> cols;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0 ; j < cols; j++)
            file >> A(i, j);

    file.close();

    return A;
}

Eigen::VectorXd readVectorFromFile(const std::string& filename) {
    std::ifstream file(filename);

    int size;

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file : " + filename + " !");
    }

    file >> size;

    Eigen::VectorXd v = Eigen::VectorXd::Zero(size);

    for (int i = 0; i < size; i++) {
        file >> v(i);
    }

    file.close();

    return v;
}

Eigen::VectorXd buildEnergyScale(double energyAxisSize, double energyAxisScale, double energyAxisOffset) {
    Eigen::VectorXd energyScale = Eigen::VectorXd::LinSpaced(energyAxisSize, energyAxisOffset, energyAxisOffset + energyAxisSize * energyAxisScale);

    return energyScale;
}

void gaussian(const Eigen::Ref<const Eigen::VectorXd>& x, double mu, double sigma, Eigen::Ref<Eigen::VectorXd> f) {
    f = (1.0 / (sigma * std::sqrt(2.0 * M_PI))) * ((-1.0) * ((x.array() - mu).array().square()) / (2.0 * std::pow(sigma, 2.0))).exp();
}

int countGColumns(const std::vector<std::string>& elements, const std::vector<std::string>& splitLinesElements) {
    std::unordered_set<std::string> splitLinesElementsSet(splitLinesElements.begin(), splitLinesElements.end());

    int totalColumns = 0;
    for (const auto& element : elements) {
        if (splitLinesElementsSet.count(element)) {
            totalColumns += 2;
        } 
        
        else {
            totalColumns += 1;
        }
    }

    return totalColumns;
}

Eigen::MatrixXd readXrayLines(const std::string& element, const nlohmann::json& xrayDBFile) {
    if (!xrayDBFile["table"].contains(element)) {
        throw std::runtime_error("Element " + element + " not found in the X-ray lines database.");
    }

    const auto& elementData = xrayDBFile["table"][element];
    size_t numLines = elementData.size();

    Eigen::MatrixXd dataMatrix = Eigen::MatrixXd::Zero(numLines, 2);

    size_t idx = 0;
    for (const auto& [line, data] : elementData.items()) {
        dataMatrix(idx, 0) = data["energy"];
        dataMatrix(idx, 1) = data["cs"];
        idx++;
    }

    return dataMatrix;
}

Eigen::MatrixXd readMassAbsorptionCoefficients(const std::string& element, const nlohmann::json& massAbsorptionCoefficientFile) {
    if (!massAbsorptionCoefficientFile["interpolated_MACs"].contains(element)) {
        throw std::runtime_error("Element " + element + " not found in the mass-absorption coefficients database.");
    }

    const auto& elementData = massAbsorptionCoefficientFile["interpolated_MACs"][element];

    const auto& energies = elementData["energies (keV)"];
    const auto& mass_absorption_coefficients = elementData["mass_absorption_coefficient (cm2/g)"];
    size_t numLines = energies.size();

    Eigen::MatrixXd dataMatrix = Eigen::MatrixXd::Zero(numLines, 2);

    dataMatrix.col(0) = Eigen::Map<Eigen::VectorXd>(energies.get<std::vector<double>>().data(), numLines);
    dataMatrix.col(1) = Eigen::Map<Eigen::VectorXd>(mass_absorption_coefficients.get<std::vector<double>>().data(), numLines);

    return dataMatrix;
}

Eigen::MatrixXd readDetectorEfficiency(const std::string& filename) {
    std::ifstream efficiencyFile(filename);

    if (!efficiencyFile.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<double> energies;
    std::vector<double> efficiencies;
    double energy, efficiency;

    while (efficiencyFile >> energy >> efficiency) {
        energies.push_back(energy);
        efficiencies.push_back(efficiency);
    }

    efficiencyFile.close();

    size_t numLines = energies.size();

    Eigen::MatrixXd dataMatrix = Eigen::MatrixXd::Zero(numLines, 2);

    dataMatrix.col(0) = Eigen::Map<Eigen::VectorXd>(energies.data(), numLines);
    dataMatrix.col(1) = Eigen::Map<Eigen::VectorXd>(efficiencies.data(), numLines);

    return dataMatrix;
}

int findClosestValueIndexInVector(const Eigen::Ref<const Eigen::VectorXd>& v, double value) {
    double minDiff = std::numeric_limits<double>::max();
    int closestIndex = -1;

    for (int i = 0; i < v.size(); i++) {
        double diff = std::fabs(v(i) - value);

        if (diff < minDiff) {
            minDiff = diff;
            closestIndex = i;
        }
    }

    return closestIndex;
}

Eigen::VectorXd lifshinBremsstrahlungB0(const Eigen::Ref<const Eigen::VectorXd>& x, double b0, double E0) {
    Eigen::VectorXd B0 = Eigen::VectorXd::Zero(x.size());

    B0 = b0 * (E0 - x.array()) / (E0 * x.array()) * (1.0 - (E0 - x.array()) / E0);

    return B0;
}

Eigen::VectorXd lifshinBremsstrahlungB1(const Eigen::Ref<const Eigen::VectorXd>& x, double b1, double E0) {
    Eigen::VectorXd B1 = Eigen::VectorXd::Zero(x.size());

    B1 = b1 * (E0 - x.array()).pow(2) / (std::pow(E0, 2) * x.array());

    return B1;
}

Eigen::VectorXd atomicToWeightPercent(const Eigen::Ref<const Eigen::VectorXd>& atomicPercentages, const std::vector<std::string>& elements, const nlohmann::json& periodicTableInfoFile) {
    Eigen::VectorXd atomicMasses = Eigen::VectorXd::Zero(atomicPercentages.size());
   
    for (int i = 0; i < atomicPercentages.size(); i++) {
        std::string element = elements[i];

        if (periodicTableInfoFile["table"].contains(element)) {
            atomicMasses(i) = periodicTableInfoFile["table"][element]["atomic_mass"];
        }

        else {
            throw std::runtime_error("Element " + element + " not found in the periodic table data file.");
        }
    }

    Eigen::VectorXd weightPercentages = atomicPercentages.cwiseProduct(atomicMasses);
    double sumAtomic = weightPercentages.sum() / 100.0;

    for (int i = 0; i < weightPercentages.size(); i++) {
        if (sumAtomic == 0) {
            weightPercentages(i) = 0.0;
        }

        else {
            weightPercentages(i) /= sumAtomic;
        }
    }

    return weightPercentages;
}

double densityOfMixture(const Eigen::Ref<const Eigen::VectorXd>& weightPercentages, const std::vector<std::string>& elements, const nlohmann::json& periodicTableInfoFile, EDXSModelConstants::meanType meanType) {
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(weightPercentages.size());

    for (int i = 0; i < weightPercentages.size(); i++) {
        std::string element = elements[i];

        if (periodicTableInfoFile["table"].contains(element)) {
            densities(i) = periodicTableInfoFile["table"][element]["density"];
        }

        else {
            throw std::runtime_error("Element " + element + " not found in the periodic table data file.");
        }
    }

    Eigen::VectorXd sumDensities = Eigen::VectorXd::Zero(weightPercentages.size());
    double density = 0.0;

    if (meanType == EDXSModelConstants::meanType::HARMONIC) {
        for (int i = 0; i < weightPercentages.size(); i++) {
            if (densities(i) != 0.0) {
                sumDensities(i) = weightPercentages(i) / densities(i);
            } 
            
            else {
                throw std::runtime_error("Density of Mixture Error : The density of one of the elements is either unknown or zero.");
            }
        }
        
        double sumDensitiesSum = sumDensities.sum();
        
        if (sumDensitiesSum != 0.0) {
            density = weightPercentages.sum() / sumDensitiesSum;
        } 
        
        else {
            density = 0.0;
        }
    } 
    
    else if (meanType == EDXSModelConstants::meanType::WEIGHTED) {
        for (int i = 0; i < weightPercentages.size(); i++) {
            sumDensities(i) = weightPercentages(i) * densities(i);
        }
        
        double sumDensitiesSum = sumDensities.sum();
        double sumWeight = weightPercentages.sum();
        
        if (sumWeight != 0.0) {
            density = sumDensitiesSum / sumWeight;
        } 
        
        else {
            density = 0.0;
        }
    } 
    
    else {
        throw std::invalid_argument("Density of Mixture Error : Invalid meanType. Valid values are HARMONIC_MEAN and WEIGHTED_MEAN.");
    }

    return density;
}

double meanAtomicNumber(const Eigen::Ref<const Eigen::VectorXd>& atomicPercentages, const std::vector<std::string>& elements, const nlohmann::json& periodicTableInfoFile) {
    Eigen::VectorXd atomicNumbers = Eigen::VectorXd::Zero(atomicPercentages.size());

    for (int i = 0; i < atomicPercentages.size(); i++) {
        std::string element = elements[i];

        if (periodicTableInfoFile["table"].contains(element)) {
            atomicNumbers(i) = periodicTableInfoFile["table"][element]["number"];
        }

        else {
            throw std::runtime_error("Element " + element + " not found in the periodic table data file.");
        }
    }

    double meanAtomicNumber = (atomicPercentages.array() * atomicNumbers.array()).sum() / 100.0;

    return meanAtomicNumber;
}

int energyToArrayIndex(double energy, double energyAxisScale, double energyAxisOffset) {
    return (int(std::round((energy - energyAxisOffset) / energyAxisScale)));
}

double _basic_simpson(const Eigen::Ref<const Eigen::VectorXd>& y, int start, int stop, const Eigen::Ref<const Eigen::VectorXd>& x, double dx) {
    double result = 0.0;

    if (x.isZero()) { 
        for (int i = start; i < stop; i += 2) {
            result += y(i) + 4.0 * y(i + 1) + y(i + 2);
        }

        result *= dx / 3.0;
    }
    
    else {
        for (int i = start; i < stop; i += 2) {
            double h0 = x(i + 1) - x(i);
            double h1 = x(i + 2) - x(i + 1);
            double hsum = h0 + h1;
            double hprod = h0 * h1;
            double h0divh1 = h0 / h1;
            double temp = hsum / 6.0 * (y(i) * (2.0 - 1.0 / h0divh1) +
                                        y(i + 1) * (hsum * hsum / hprod) +
                                        y(i + 2) * (2.0 - h0divh1));
            result += temp;
        }
    }

    return result;
}

double simpson(const Eigen::Ref<const Eigen::VectorXd>& y, const Eigen::Ref<const Eigen::VectorXd>& x, double dx) {
    int N = y.size();

    if (N < 2) {
        throw std::range_error("EELS Data Integration Error : There must be at least two data points.");
    }

    double result = 0.0;

    if (N % 2 == 0) {

        if (N == 2) {
            double last_dx = dx;
            
            if (!x.isZero()) {
                last_dx = x(N - 1) - x(N - 2);
            }

            result += 0.5 * last_dx * (y(N - 1) + y(N - 2));
        } 
        
        else {
            result = _basic_simpson(y, 0, N - 3, x, dx);

            double h0 = dx, h1 = dx;

            if (!x.isZero()) {
                h0 = x(N - 2) - x(N - 3);
                h1 = x(N - 1) - x(N - 2);
            }

            double alpha = (2 * h1 * h1 + 3 * h0 * h1) / (6 * (h1 + h0));
            double beta = (h1 * h1 + 3.0 * h0 * h1) / (6 * h0);
            double eta = h1 * h1 * h1 / (6 * h0 * (h0 + h1));

            result += alpha * y(N - 1) + beta * y(N - 2) - eta * y(N - 3);
        }
    } 
    
    else {
        result = _basic_simpson(y, 0, N - 2, x, dx);
    }

    return result;
}

double F(double electronEnergy) {
    return ((1.0 + electronEnergy / 1022) / std::pow((1.0 + electronEnergy / 511), 2));
}

double thetaE(double density, double electronEnergy) {
    return (5.5 * std::pow(density, 0.3) / (F(electronEnergy) * electronEnergy));
}

double electronInelasticMeanFreePath(double density, double electronEnergy) {
    double thetaC = 20.0;
    double invLambda = 11.0 * std::pow(density, 0.3) / (200.0 * F(electronEnergy) * electronEnergy) * std::log(std::pow(thetaC, 2) / std::pow(thetaE(density, electronEnergy), 2));
    
    return (1.0 / invLambda);
}

double electronInelasticMeanFreePath(double meanAtomicNumber, double electronEnergy, double beta) {
    double meanEnergyLoss = 7.6 * std::pow(meanAtomicNumber, 0.36);
    double lambda = (106.0 * F(electronEnergy) * electronEnergy) / (meanEnergyLoss * std::log(2.0 * beta * electronEnergy / meanEnergyLoss));

    return lambda;
}

double angularCorrection(double density, double electronEnergy, double alpha, double beta) {
    double thetaC = 20.0;
    double A = std::pow(alpha, 2) + std::pow(beta, 2) + 2 * std::pow(thetaE(density, electronEnergy), 2) + std::fabs(std::pow(alpha, 2) - std::pow(beta, 2));
    double B = std::pow(alpha, 2) + std::pow(beta, 2) + 2 * std::pow(thetaC, 2) + std::fabs(std::pow(alpha, 2) - std::pow(beta, 2));
    double correction = std::log(std::pow(thetaC, 2) / std::pow(thetaE(density, electronEnergy), 2)) / std::log(A * std::pow(thetaC, 2) / B / std::pow(thetaE(density, electronEnergy), 2));

    return correction;
}

double estimateThicknessAtPixel(const Eigen::Ref<const Eigen::VectorXd>& EELSLowLossSpectrum, const Eigen::Ref<const Eigen::VectorXd>& energyAxis, double energyAxisScale, double energyAxisOffset, double zeroLossPeakThreshold, double density, double electronEnergy, double alpha, double beta) {
    double totalIntensity = simpson(EELSLowLossSpectrum, energyAxis);
    int thresholdIndex = energyToArrayIndex(zeroLossPeakThreshold, energyAxisScale, energyAxisOffset);
    double zeroLossPeakIntensity = simpson(EELSLowLossSpectrum(Eigen::seq(0, thresholdIndex)), energyAxis(Eigen::seq(0, thresholdIndex)));

    double tOverLambda = std::log(totalIntensity / zeroLossPeakIntensity);
    double lambdaAngularCorrection = angularCorrection(density, electronEnergy, alpha, beta);
    double lambda = electronInelasticMeanFreePath(density, electronEnergy);

    return (tOverLambda * lambdaAngularCorrection * lambda);
}

double estimateThicknessAtPixel(const Eigen::Ref<const Eigen::VectorXd>& EELSLowLossSpectrum, const Eigen::Ref<const Eigen::VectorXd>& energyAxis, double energyAxisScale, double energyAxisOffset, double zeroLossPeakThreshold, double meanAtomicNumber, double electronEnergy, double beta) {
    double totalIntensity = simpson(EELSLowLossSpectrum, energyAxis);
    int thresholdIndex = energyToArrayIndex(zeroLossPeakThreshold, energyAxisScale, energyAxisOffset);
    double zeroLossPeakIntensity = simpson(EELSLowLossSpectrum(Eigen::seq(0, thresholdIndex)), energyAxis(Eigen::seq(0, thresholdIndex)));

    double tOverLambda = std::log(totalIntensity / zeroLossPeakIntensity);
    double lambda = electronInelasticMeanFreePath(meanAtomicNumber, electronEnergy, beta);

    return (tOverLambda * lambda);
}

Eigen::VectorXd computeMassAbsorptionCoefficients(const Eigen::Ref<const Eigen::VectorXd>& energyRange, const std::vector<std::string>& elements, const Eigen::Ref<const Eigen::VectorXd>& concentrations, bool atomicFraction, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile) {
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(energyRange.size());
    Eigen::VectorXd weightPercentage = Eigen::VectorXd::Zero(elements.size());

    if (elements.size() == 0 || concentrations.size() == 0) {
        return (1 / energyRange.array().pow(3));
    }
    
    if (atomicFraction) {
        weightPercentage = atomicToWeightPercent(concentrations, elements, periodicTableInfoFile).array() / 100.0;
    } 
    
    else {
        weightPercentage = concentrations.array() / 100.0;
    }

    double sumElements = weightPercentage.sum();

    for (size_t i = 0; i < elements.size(); i++) {
        Eigen::MatrixXd massAbsorptionCoefficients = readMassAbsorptionCoefficients(elements[i], massAbsorptionCoefficientFile);

        Eigen::VectorXd muElement = Eigen::VectorXd::Zero(energyRange.size());

        for (int j = 0; j < energyRange.size(); j++) {
            int idx = findClosestValueIndexInVector(massAbsorptionCoefficients.col(0), energyRange(j));
            muElement(j) = massAbsorptionCoefficients(idx, 1) * weightPercentage(i) / sumElements;
        }

        mu += muElement;
    }

    return mu;
}

double computeSingleMassAbsorptionCoefficient(double energy, const std::string& element, const nlohmann::json& massAbsorptionCoefficientFile) {
    Eigen::MatrixXd massAbsorptionCoefficients = readMassAbsorptionCoefficients(element, massAbsorptionCoefficientFile);

    int idx = findClosestValueIndexInVector(massAbsorptionCoefficients.col(0), energy);

    return massAbsorptionCoefficients(idx, 1);
}

Eigen::VectorXd computeAbsorptionCorrection(const Eigen::Ref<const Eigen::VectorXd>& energyRange, const std::vector<std::string>& elements, const Eigen::Ref<const Eigen::VectorXd>& concentrations, double thickness, double takeOffAngle, double* density, bool atomicFraction, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile) {
    if (thickness == 0) {
        return Eigen::VectorXd::Constant(energyRange.size(), 1.0);
    }
    
    Eigen::VectorXd mu = computeMassAbsorptionCoefficients(energyRange, elements, concentrations, atomicFraction, periodicTableInfoFile, massAbsorptionCoefficientFile);

    double toaRad = takeOffAngle * M_PI / 180.0;

    Eigen::VectorXd weightPercentage = Eigen::VectorXd::Zero(elements.size());

    if (atomicFraction) {
        weightPercentage = atomicToWeightPercent(concentrations, elements, periodicTableInfoFile).array() / 100.0;
    } 
    
    else {
        weightPercentage = concentrations.array() / 100.0;
    }

    if (*density == 0.0) {
        *density = densityOfMixture(weightPercentage.array() * 100.0, elements, periodicTableInfoFile, EDXSModelConstants::meanType::HARMONIC);
    }

    Eigen::VectorXd chi = mu.array() * *density * thickness / std::sin(toaRad);

    return ((1.0 - ((-1.0) * chi.array()).exp()) / chi.array());
}

double computeSingleAbsorptionCorrection(double energy, const std::string& element, double thickness, double takeOffAngle, double* density, const nlohmann::json& periodicTableInfoFile, const nlohmann::json& massAbsorptionCoefficientFile) {
    if (thickness == 0) {
        return 1.0;
    }

    double mu = computeSingleMassAbsorptionCoefficient(energy, element, massAbsorptionCoefficientFile);

    double toaRad = takeOffAngle * M_PI / 180.0;

    if (periodicTableInfoFile["table"].contains(element)) {
        *density = periodicTableInfoFile["table"][element]["density"];
    }
        
    else {
        throw std::runtime_error("Element " + element + " not found in the periodic table data file.");
    }

    double chi = mu * *density * thickness / std::sin(toaRad);

    return ((1.0 - std::exp(chi * (-1.0))) / chi);
}

Eigen::MatrixXd getExplainedIntensity(const Eigen::Ref<const Eigen::MatrixXd>& G, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H) {
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(W.rows(), W.cols());
    
    for (int i = 0; i < W.rows(); i++) {
        for (int j = 0; j < W.cols(); j++) {
            N(i, j) = (G.col(i) * H.row(j)).sum() * W(i, j);
        }
    }

    return N;
}

void printConcentrationReport(const Eigen::Ref<const Eigen::MatrixXd>& G, const Eigen::Ref<const Eigen::MatrixXd>& W, const Eigen::Ref<const Eigen::MatrixXd>& H, const std::vector<std::string>& modelElements, const std::vector<std::string>& selectedElements, bool fitError) {
    Eigen::MatrixXd N = getExplainedIntensity(G, W, H);
    Eigen::MatrixXd sqrtN = N.array().sqrt();

    Eigen::MatrixXd percentages = Eigen::MatrixXd::Zero(W.rows(), W.cols());
    percentages = sqrtN.array() / N.array() * 100.0;

    std::vector<std::string> quantElements;
    std::vector<int> quantIndices;
    std::regex pattern_low(R"(^[A-Za-z0-9]+_low$)");
    std::regex pattern_high(R"(^([A-Za-z0-9]+)_high$)");

    if (!selectedElements.empty()) {
        for (size_t i = 0; i < selectedElements.size(); i++) {
            for (size_t j = 0; j < modelElements.size(); j++) {
                std::string element;
                
                if (std::regex_match(modelElements[j], pattern_low)) {
                    continue;
                }

                else if (std::regex_match(modelElements[j], pattern_high)) {
                    std::smatch match;
                    std::regex_search(modelElements[j], match, pattern_high);
                    element = match.str(1);
                }

                else {
                    element = modelElements[j];
                }

                if (element == selectedElements[i]) {
                    quantIndices.push_back(j);
                    quantElements.push_back(element);
                }
            }
        }
    }

    else {
        for (size_t i = 0; i < modelElements.size(); i++) {
            if (std::regex_match(modelElements[i], pattern_low)) {
                continue;
            }

            else if (std::regex_match(modelElements[i], pattern_high)) {
                std::smatch match;
                std::regex_search(modelElements[i], match, pattern_high);
                quantElements.push_back(match.str(1));
                quantIndices.push_back(i);
            }

            else {
                quantElements.push_back(modelElements[i]);
                quantIndices.push_back(i);
            }
        }
    }

    Eigen::MatrixXd WQuant = Eigen::MatrixXd::Zero(quantIndices.size(), W.cols());
    WQuant = W(quantIndices, Eigen::placeholders::all);
    Eigen::VectorXd sumWQuant = WQuant.colwise().sum().array();

    WQuant = WQuant.array() / sumWQuant.transpose().replicate(WQuant.rows(), 1).array() * 100.0;

    Eigen::MatrixXd errors = Eigen::MatrixXd::Zero(WQuant.rows(), WQuant.cols());

    if (fitError) {
        errors = percentages(quantIndices, Eigen::placeholders::all);
    }

    std::vector<std::string> field_list = {"Elements"};

    for (int i = 0; i < W.cols(); i++) {
        field_list.push_back("Phase " + std::to_string(i) + " (at.%)");
        
        if (fitError) {
            field_list.push_back("Phase " + std::to_string(i) + " std (%)");
        }
    }

    const int columnWidth = 20;
    const std::string separator = "+";
    const std::string line = std::string(columnWidth, '-');
    
    for (size_t i = 0; i < field_list.size(); i++) {
        std::cout << separator << line;
    }

    std::cout << separator << "\n";

    for (const auto& field : field_list) {
        std::cout << "|" << std::setw(columnWidth) << std::left << field;
    }

    std::cout << "|\n";

    for (size_t i = 0; i < field_list.size(); i++) {
        std::cout << separator << line;
    }

    std::cout << separator << "\n";

    for (size_t i = 0; i < quantElements.size(); i++) {
        std::cout << "|" << std::setw(columnWidth) << std::left << quantElements[i];
        
        for (int j = 0; j < W.cols(); j++) {
            std::cout << "|" << std::setw(columnWidth) << std::right << std::fixed << std::setprecision(3) << WQuant(i, j);
            
            if (fitError) {
                std::cout << "|" << std::setw(columnWidth) << std::right << std::fixed << std::setprecision(3) << errors(i, j);
            }
        }
        
        std::cout << "|\n";

        for (size_t k = 0; k < field_list.size(); k++) {
            std::cout << separator << line;
        }

        std::cout << separator << "\n";
    }
}