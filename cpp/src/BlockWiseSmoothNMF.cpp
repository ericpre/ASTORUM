#include "../include/BlockWiseSmoothNMF.h"

double euclideanDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return (a - b).norm();
}

double manhattanDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return (a - b).lpNorm<1>();
}

double mahalanobisDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b, const Eigen::MatrixXd& inverseCovarianceMatrix) {
    Eigen::VectorXd diff = a - b;
    return std::sqrt(diff.transpose() * inverseCovarianceMatrix * diff);
}

double minkowskiDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b, int p) {
    switch (p) {
        case 1:
            return manhattanDistance(a, b);
        
        case 2:
            return euclideanDistance(a, b);
        
        case 3:
            return (a - b).lpNorm<3>();

        case 4:
            return (a - b).lpNorm<4>();

        case 5:
            return (a - b).lpNorm<5>();

        case 6:
            return (a - b).lpNorm<6>();
        
        case 7:
            return (a - b).lpNorm<7>();

        case 8:
            return (a - b).lpNorm<8>();

        case 9:
            return (a - b).lpNorm<9>();

        case 10:
            return (a - b).lpNorm<10>();

        default:
            throw std::invalid_argument("Estimator Error : Invalid norm order for Minkowski distance.");
        }
}

double cosineDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return 1.0 - a.dot(b) / (a.norm() * b.norm());
}

double chebyshevDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return (a - b).lpNorm<Eigen::Infinity>();
}

double hammingDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return (a.array() != b.array()).cast<double>().sum();
}

double jaccardDistance(const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
    return 1.0 - (a.array().min(b.array())).sum() / (a.array().max(b.array())).sum();
}


BlockWiseSmoothNMF::BlockWiseSmoothNMF() {
    _blocks = 0;
    _Gcols = 0;
    _nPoints = 0;
    _nDims = 0;
    _nClusters = 0;
};


BlockWiseSmoothNMF::BlockWiseSmoothNMF(
        const std::string& inputDir,
        const std::string& outputDir,
        int blocks,
        int blockWidth,
        int blockHeight,
        int nClusters,
        const Eigen::Ref<const Eigen::VectorXi>& componentsVector
        )
    {
        _inputDir = inputDir;
        _outputDir = outputDir;
        _blocks = blocks;
        _blockWidth = blockWidth;
        _blockHeight = blockHeight;
        _nClusters = nClusters;
        _componentsVector = componentsVector;

        _nPoints = _componentsVector.sum();
        _pixels = _blockWidth * _blockHeight;
};


BlockWiseSmoothNMF::~BlockWiseSmoothNMF() {
};


void BlockWiseSmoothNMF::initialiseModel(
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
        _model = EDXSDataset(beamEnergy, problemType, absorptionModelType, azimuthAngle, elevationAngle, tiltStage, thickness, density, widthSlope, widthIntercept, energyAxisSize, energyAxisScale, energyAxisOffset, detectorEfficiency, xrayDB, massAbsorptionCoefficientsFilePath, decompositionResultsFilePath, absorptionMatrixFilePath, thicknessMapFilePath, periodicTableInfoFilePath, elements, splitLinesElements, energyThresholds, quantificationElements, absorptionElements, absorptionElementsConcentrations);
        
        _model.generateGMatrix();
        _G = _model._G;
        _Gcols = _G.cols();
        _nDims = _Gcols;

        if (_model._problemType == EDXSModelConstants::problemType::BREMSSTRAHLUNG) {
            _nDims -= 2;
        }

        else {
            throw std::invalid_argument("Estimator Error : BlockWiseSmoothNMF currently supports only BREMSSTRAHLUNG problem type.");
        }

        _WElements = Eigen::MatrixXd::Zero(_nPoints, _nDims);
        _WBremsstrahlung = Eigen::MatrixXd::Zero(_nPoints, 2);
        _WFull = Eigen::MatrixXd::Zero(_nPoints, _Gcols);
        _WClustered = Eigen::MatrixXd::Zero(_Gcols, _nClusters);

        if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
            _GB0 = Eigen::MatrixXd::Zero(_model._energyAxisSize, _blocks);
            _GB1 = Eigen::MatrixXd::Zero(_model._energyAxisSize, _blocks);
        }

        else {
            _GB0 = Eigen::MatrixXd::Zero(0, 0);
            _GB1 = Eigen::MatrixXd::Zero(0, 0);
        }
}

void BlockWiseSmoothNMF::setModel(EDXSDataset model) {
    _model = model;
    _G = _model._G;
    _Gcols = _G.cols();
    _nDims = _Gcols;

    if (_model._problemType == EDXSModelConstants::problemType::BREMSSTRAHLUNG) {
        _nDims -= 2;
    }

    else {
        throw std::invalid_argument("Estimator Error : BlockWiseSmoothNMF currently supports only BREMSSTRAHLUNG problem type.");
    }

    _WElements = Eigen::MatrixXd::Zero(_nPoints, _nDims);
    _WBremsstrahlung = Eigen::MatrixXd::Zero(_nPoints, 2);
    _WFull = Eigen::MatrixXd::Zero(_nPoints, _Gcols);
    _WClustered = Eigen::MatrixXd::Zero(_Gcols, _nClusters);


    if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
        _GB0 = Eigen::MatrixXd::Zero(_model._energyAxisSize, _blocks);
        _GB1 = Eigen::MatrixXd::Zero(_model._energyAxisSize, _blocks);
    }

    else {
        _GB0 = Eigen::MatrixXd::Zero(0, 0);
        _GB1 = Eigen::MatrixXd::Zero(0, 0);
    }
}

Eigen::MatrixXd BlockWiseSmoothNMF::postProcessData(const Eigen::Ref<const Eigen::MatrixXd>& W, int separationOrder) {
    Eigen::MatrixXd WMod = W;

    for (int j = 0; j < WMod.cols(); j++) {
        for (int i = 0; i < WMod.rows() - 2; i++) {
            if (WMod(i, j) <= 1E-3) {
                WMod(i, j) = 0.0;
            }
            if (WMod(i, j) >= 0.98) {
                WMod(i, j) = 1.0;
            }
        }
    }

    if (separationOrder >= 1) {
        int oneIndexRow = -1;
        int oneIndexCol = -1;

        for (int j = 0; j < WMod.cols(); j++) {
            for (int i = 0; i < WMod.rows() - 2; i++) {
                if (WMod(i, j) <= 1E-3) {
                    WMod(i, j) = 0.0;
                }

                if (WMod(i, j) >= 0.98) {
                    WMod(i, j) = 1.0;
                    oneIndexRow = i;
                    oneIndexCol = j;
                }

                if (WMod(i, j) == 1.0) {
                    oneIndexRow = i;
                    oneIndexCol = j;
                }
            }
            
            if (oneIndexRow != -1 && oneIndexCol != -1) {
                for (int i = 0; i < WMod.rows() - 2; i++) {
                    if (i != oneIndexRow)
                        WMod(i, oneIndexCol) = 0.0;
                }
                
                for (int i = 0; i < WMod.cols(); i++) {
                    if (i != oneIndexCol)
                        WMod(oneIndexRow, i) = 0.0;
                }
            }
        }

        for (int i = 0; i < WMod.cols(); ++i) {
            double colSum = WMod.col(i).sum();
            if (colSum != 0.0)
                WMod.col(i) /= colSum;
        }
    }


    if (separationOrder == 2) {
        for (int j = 0; j < WMod.cols(); j++) {
            int secondOrderRowOne = -1;
            int secondOrderRowTwo = -1;
            int secondOrderCol = -1;

            for (int i = 0; i < WMod.rows() - 2; i++) {
                for (int k = i + 1; k < WMod.rows() - 2; k++) {
                    if (WMod(i, j) <= 0.95 && WMod(k, j) <= 0.95) {
                        double sum = WMod(i, j) + WMod(k, j);
                        if (sum >= 0.95 && sum <= 1.05) {
                            secondOrderCol = j;
                            secondOrderRowOne = i;
                            secondOrderRowTwo = k;
                        }
                    }
                }
            }

            if (secondOrderRowOne != -1 && secondOrderRowTwo != -1 && secondOrderCol != -1) {
                for (int i = 0; i < WMod.rows() - 2; i++) {
                    if (i != secondOrderRowOne && i != secondOrderRowTwo)
                        WMod(i, secondOrderCol) = 0.0;
                }
                
                for (int i = 0; i < WMod.cols(); i++) {
                    if (i != secondOrderCol) {
                        WMod(secondOrderRowOne, i) = 0.0;
                        WMod(secondOrderRowTwo, i) = 0.0;
                    }
                }
            }
        }
    }

    for (int i = 0; i < WMod.cols(); i++) {
        double colSum = WMod.col(i).sum();
        
        if (colSum != 0.0)
            WMod.col(i) /= colSum;
    }

    return WMod;
}

void BlockWiseSmoothNMF::computeWBlocks( 
        const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
        SmoothNMFConstants::initialisation init, 
        int maxIter, 
        int randomSeed, 
        SmoothNMFConstants::algorithm algorithm,
        double tol, 
        double logShift, 
        double eps, 
        double lambdaL, 
        double mu, 
        double epsilonReg, 
        double dichotomyTol, 
        double sigmaL, 
        double gammaStepScalar,
        bool simplexW, 
        bool simplexH, 
        bool l2, 
        bool verbose, 
        bool safe,
        bool debug, 
        bool normalise, 
        bool noStopCriterion, 
        bool lineSearch,
        const Eigen::Ref<const Eigen::VectorXi>& separationOrder,
        bool precomputedW,
        bool writeWblocks
        )
    {
        std::cout<<"Computing W blocks..."<<"\n\n";

        int comp = 0;

        if (precomputedW) {
            for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
                std::string WFilename = _outputDir + "/W/W_block_" + std::to_string(blockIndex) + ".onmf";
                Eigen::MatrixXd WBlock = readMatrixFromFile(WFilename);

                if (WBlock.rows() != _Gcols || WBlock.cols() != _componentsVector(blockIndex)) {
                    throw std::invalid_argument("Estimator Error : Pre-computed W block dimensions do not match the expected values.");
                }

                WBlock(Eigen::seq(0, WBlock.rows() - 3), Eigen::placeholders::all) = postProcessData(WBlock(Eigen::seq(0, WBlock.rows() - 3), Eigen::placeholders::all), separationOrder(blockIndex));

                for(int i = 0; i < WBlock.cols(); i++) {
                    _WElements(comp, Eigen::placeholders::all) = WBlock(Eigen::seq(0, WBlock.rows() - 3), i).transpose();
                    _WBremsstrahlung.row(comp) = WBlock(Eigen::placeholders::lastN(2), i).transpose();
                    comp++;
                }

                std::cout<<"Loaded pre-computed W block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";
            }

            std::string GFilename = _outputDir + "/G.onmf";
            _G = readMatrixFromFile(GFilename);
            _model._G = _G;
        }

        else {
            for (int blockIndex = 0; blockIndex < _blocks; blockIndex++){
                std::string XFilename = _inputDir + "/X/X_block_" + std::to_string(blockIndex) + ".inmf";
                Eigen::MatrixXd X = readMatrixFromFile(XFilename);

                if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
                    std::string AFilename = _outputDir + "/A/A_block_" + std::to_string(blockIndex) + ".onmf";
                    _model._absorptionMatrixFilePath = AFilename;
                }

                Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_Gcols, _componentsVector(blockIndex));
                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_componentsVector(blockIndex), _pixels);

                Eigen::MatrixXd fixedW = Eigen::MatrixXd::Constant(_Gcols, _componentsVector(blockIndex), -1.0);
                Eigen::MatrixXd fixedH = Eigen::MatrixXd::Constant(_componentsVector(blockIndex), _pixels, -1.0);
                
                SmoothNMF snmf = SmoothNMF(X, W, H, fixedW, fixedH, gammaStepArray, 
                                        _componentsVector(blockIndex), init, maxIter, randomSeed, algorithm, 
                                        tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                        simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch);

                snmf.setModel(_model);
                snmf.fitTransform();

                if (writeWblocks) {
                    std::string WFilename = _outputDir + "/W/W_block_" + std::to_string(blockIndex) + ".onmf";
                    writeMatrixToFile(snmf._W, WFilename);
                }

                snmf._W(Eigen::seq(0, snmf._W.rows() - 3), Eigen::placeholders::all) = postProcessData(snmf._W(Eigen::seq(0, snmf._W.rows() - 3), Eigen::placeholders::all), separationOrder(blockIndex));

                for(int i = 0; i < snmf._W.cols(); i++) {
                    _WElements(comp, Eigen::placeholders::all) = snmf._W(Eigen::seq(0, snmf._W.rows() - 3), i).transpose();
                    _WBremsstrahlung.row(comp) = snmf._W(Eigen::placeholders::lastN(2), i).transpose();
                    comp++;
                }

                if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
                    _GB0.col(blockIndex) = snmf._G.col(snmf._G.cols() - 2);
                    _GB1.col(blockIndex) = snmf._G.col(snmf._G.cols() - 1);
                }
                
                std::cout<<"Computed W block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";
            }

            if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
                Eigen::VectorXd GB0Avg = _GB0.rowwise().sum().array() / _blocks;
                Eigen::VectorXd GB1Avg = _GB1.rowwise().sum().array() / _blocks;
                _G.col(_G.cols() - 2) = GB0Avg;
                _G.col(_G.cols() - 1) = GB1Avg;
                _model._G = _G;
            }
            
            std::string GFilename = _outputDir + "/G.onmf";
            writeMatrixToFile(_G, GFilename);
        }

        _WFull << _WElements, _WBremsstrahlung;
}

void BlockWiseSmoothNMF::KmeansClustering(const Eigen::Ref<const Eigen::MatrixXd>& data, int nIter, double tolerance) {
    std::cout<<"Performing K-means clustering..."<<"\n";

    int nDims = data.cols();
    int nPoints = data.rows();

    Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(_nClusters, nDims);
    Eigen::MatrixXd oldCentroids;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    int firstCentroidIndex = static_cast<int>(dis(gen) * nPoints);
    centroids.row(0) = data.row(firstCentroidIndex);

    for (int i = 1; i < _nClusters; i++) {
        Eigen::VectorXd minDistances = (data.rowwise() - centroids.row(0)).rowwise().squaredNorm();

        for (int j = 1; j < i; j++) {
            Eigen::VectorXd distances = (data.rowwise() - centroids.row(j)).rowwise().squaredNorm();
            minDistances = minDistances.cwiseMin(distances);
        }

        double totalDist = minDistances.sum();
        std::vector<double> probabilities(nPoints);
        
        for (int j = 0; j < nPoints; j++) {
            probabilities[j] = minDistances[j] / totalDist;
        }

        std::discrete_distribution<> d(probabilities.begin(), probabilities.end());
        int nextCentroidIndex = d(gen);
        centroids.row(i) = data.row(nextCentroidIndex);
    }

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(_nClusters, _nClusters);
    Eigen::MatrixXd post(nPoints, _nClusters);
    Eigen::VectorXd minVals(nPoints);

    double oldE = 0;

    for (int iter = 0; iter < nIter; iter++) {
        oldCentroids = centroids;

        Eigen::MatrixXd distances(nPoints, _nClusters);

        for (int i = 0; i < _nClusters; i++) {
            for (int j = 0; j < nPoints; j++) {
                distances(j, i) = (data.row(j) - centroids.row(i)).squaredNorm();
            }
        }

        int r, c;

        for (int i = 0; i < nPoints; i++) {
            minVals(i) = distances.row(i).minCoeff(&r, &c);
            post.row(i) = I.row(c);
        }

        Eigen::VectorXd sumPost = post.colwise().sum();

        for (int i = 0; i < _nClusters; i++) {
            if (sumPost(i) > 0) {
                Eigen::MatrixXd S = Eigen::MatrixXd::Zero(1, nDims);

                for (int j = 0; j < nPoints; j++) {
                    if (post(j, i) == 1) {
                        S += data.row(j);
                    }
                }

                centroids.row(i) = S / sumPost(i);
            }
        }

        double E = minVals.sum();
        double eDiff = std::fabs(oldE - E);
        double cDiff = (centroids - oldCentroids).cwiseAbs().maxCoeff();

        if (iter > 1) {
            if (cDiff < tolerance && eDiff < tolerance) {
                break;
            }
        }

        oldE = E;
    }

    for (int i =0; i < _nClusters; i ++) {
        std::multimap<double, int> cluster;

        for (int j = 0; j < nPoints; j++) {
            if (post(j, i) == 1) {
                cluster.insert(std::make_pair(minVals(j), j));
            }
        }

        std::vector<int> items;

        for (std::multimap<double, int>::iterator it = cluster.begin(); it != cluster.end(); it++) {
            items.push_back(it->second);
        }

        _clusterLabels.push_back(items);
    }
}

void BlockWiseSmoothNMF::spectralClustering(double sigma, BlockWiseSmoothNMFConstants::distanceMetric metric, int p, int nIter, double tolerance) {
    std::cout<<"Performing spectral clustering..."<<"\n";

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(_nPoints, _nPoints);

    for (int i = 0; i < _nPoints; i++) {
        for (int j = i + 1; j < _nPoints; j++) {
            double distance = 0.0;

            switch (metric) {
                case BlockWiseSmoothNMFConstants::distanceMetric::EUCLIDEAN:
                    distance = euclideanDistance(_WElements.row(i), _WElements.row(j));
                    break;

                case BlockWiseSmoothNMFConstants::distanceMetric::MANHATTAN:
                    distance = manhattanDistance(_WElements.row(i), _WElements.row(j));
                    break;

                case BlockWiseSmoothNMFConstants::distanceMetric::MAHALANOBIS:
                    // Not yet implemented
                    break;
                
                case BlockWiseSmoothNMFConstants::distanceMetric::MINKOWSKI:
                    distance = minkowskiDistance(_WElements.row(i), _WElements.row(j), p);
                    break;
                
                case BlockWiseSmoothNMFConstants::distanceMetric::COSINE:
                    distance = cosineDistance(_WElements.row(i), _WElements.row(j));
                    break;

                case BlockWiseSmoothNMFConstants::distanceMetric::CHEBYSHEV:
                    distance = chebyshevDistance(_WElements.row(i), _WElements.row(j));
                    break;

                case BlockWiseSmoothNMFConstants::distanceMetric::HAMMING:
                    distance = hammingDistance(_WElements.row(i), _WElements.row(j));
                    break;

                case BlockWiseSmoothNMFConstants::distanceMetric::JACCARD:
                    distance = jaccardDistance(_WElements.row(i), _WElements.row(j));
                    break;
                
                default:
                    throw std::invalid_argument("Estimator Error : Invalid distance metric for spectral clustering.");
            }

            double similarityValue = std::exp((-1.0) * std::pow(distance, 2) / (2 * std::pow(sigma, 2)));
            S(i, j) = similarityValue;
            S(j, i) = similarityValue;
        }

        S(i, i) = 1.0;
    }

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(_nPoints, _nPoints);

    Eigen::VectorXd degrees = S.rowwise().sum();
    Eigen::VectorXd degreesInvSqrt = degrees.array().sqrt().cwiseInverse();
    Eigen::MatrixXd DInvSqrt = degreesInvSqrt.asDiagonal();

    L = Eigen::MatrixXd::Identity(_nPoints, _nPoints) - (DInvSqrt * S * DInvSqrt);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();

    Eigen::MatrixXd features = eigenvectors.block(0, 0, eigenvectors.rows(), _nClusters);

    KmeansClustering(features, nIter, tolerance);
}

void BlockWiseSmoothNMF::printClusterLabels() {
    for (int i = 0; i < int(_clusterLabels.size()); i++) {
        std::cout << "Cluster (K-means) " << i << " : Item(s) : ";
        std::copy(_clusterLabels[i].begin(), _clusterLabels[i].end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout<<"\n";
    }
}

void BlockWiseSmoothNMF::computeClusteredW() {
    std::cout<<"Computing final W..."<<"\n";

    for (int i = 0; i < _nClusters; i++) {
        const std::vector<int>& clusterIndices = _clusterLabels[i];
        int clusterSize = clusterIndices.size();
        
        Eigen::RowVectorXd centroid = Eigen::RowVectorXd::Zero(_Gcols);
        
        for (int j = 0; j < clusterSize; j++) {
            centroid += _WFull.row(clusterIndices[j]);
        }

        centroid /= clusterSize;

        _WClustered.col(i) = centroid.transpose();
    }

    std::string WFilename = _outputDir + "/W/W_clustered.onmf";

    writeMatrixToFile(_WClustered, WFilename);
}

void BlockWiseSmoothNMF::computeComponentPresences() {
    int startIndex = 0;

    _componentPresences = Eigen::MatrixXi::Zero(_blocks, _nClusters);

    for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
        int numComponents = _componentsVector[blockIndex];

        for (int compIndex = 0; compIndex < numComponents; compIndex++) {
            int globalCompIndex = startIndex + compIndex;
            int clusterIndex = -1;

            for (int clustIdx = 0; clustIdx < _nClusters; clustIdx++) {
                const std::vector<int>& clusterIndices = _clusterLabels[clustIdx];
                for (int idx : clusterIndices) {
                    if (idx == globalCompIndex) {
                        clusterIndex = clustIdx;
                        break;
                    }
                }

                if (clusterIndex != -1) {
                    break;
                }
            }

            if (clusterIndex != -1) {
                _componentPresences(blockIndex, clusterIndex) = 1;
            }
        }

        startIndex += numComponents;
    }
}

void BlockWiseSmoothNMF::computeHBlocks( 
        const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
        SmoothNMFConstants::initialisation init, 
        int maxIter, 
        int randomSeed, 
        SmoothNMFConstants::algorithm algorithm,
        double tol, 
        double logShift, 
        double eps, 
        double lambdaL, 
        double mu, 
        double epsilonReg, 
        double dichotomyTol, 
        double sigmaL, 
        double gammaStepScalar,
        bool simplexW, 
        bool simplexH, 
        bool l2, 
        bool verbose, 
        bool safe,
        bool debug, 
        bool normalise, 
        bool noStopCriterion, 
        bool lineSearch
        )
    {
        std::cout<<"Computing H blocks (with fixed W)..."<<"\n\n";

        computeComponentPresences();

        for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
            std::string XFilename = _inputDir + "/X/X_block_" + std::to_string(blockIndex) + ".inmf";
            Eigen::MatrixXd X = readMatrixFromFile(XFilename);

            if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
                std::string AFilename = _outputDir + "/A/A_block_" + std::to_string(blockIndex) + ".onmf";
                _model._absorptionMatrixFilePath = AFilename;
            }

            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_componentsVector(blockIndex), _pixels);
            Eigen::MatrixXd fixedH = Eigen::MatrixXd::Constant(_componentsVector(blockIndex), _pixels, -1.0);
            Eigen::MatrixXd WBlock = Eigen::MatrixXd::Zero(_WClustered.rows(), 0);

            for (int j = 0; j < _nClusters; j++) {
                if (_componentPresences(blockIndex, j) == 1) {
                    WBlock.conservativeResize(Eigen::NoChange, WBlock.cols() + 1);
                    WBlock.col(WBlock.cols() - 1) = _WClustered.col(j);
                }
            }
            
            SmoothNMF snmf = SmoothNMF(X, WBlock, H, WBlock, fixedH, gammaStepArray, 
                                    _componentsVector(blockIndex), init, maxIter, randomSeed, algorithm, 
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch);

            snmf.setModel(_model);
            snmf.fitTransform();

            Eigen::MatrixXd finalH = Eigen::MatrixXd::Zero(_nClusters, _pixels);
            
            int HIndex = 0;
            
            for (int j = 0; j < _nClusters; j++) {
                if (_componentPresences(blockIndex, j) == 1) {
                    finalH.row(j) = snmf._H.row(HIndex);
                    HIndex++;
                }
            }

            std::cout<<"Computed H block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";

            std::string HFilename = _outputDir + "/H/H_block_" + std::to_string(blockIndex) + ".onmf";
            writeMatrixToFile(finalH, HFilename);

            std::string QFilename = _outputDir + "/Q/Q_block_" + std::to_string(blockIndex) + ".onmf";
            Eigen::MatrixXd Q;
            
            if (!_model._quantificationElements.empty()) {
                Q = _model.generateQuantificationMatrix(_WClustered, finalH, _model._quantificationElements);
            }

            else {
                Q = _model.generateQuantificationMatrix(_WClustered, finalH);
            }

            writeMatrixToFile(Q, QFilename);
        }
}

void BlockWiseSmoothNMF::computeHBlocksSVD() {
    std::cout<<"Computing H blocks (using SVD)..."<<"\n\n";

    computeComponentPresences();

    for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
        std::string XFilename = _inputDir + "/X/X_block_" + std::to_string(blockIndex) + ".inmf";
        Eigen::MatrixXd X = readMatrixFromFile(XFilename);

        if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
            std::string AFilename = _outputDir + "/A/A_block_" + std::to_string(blockIndex) + ".onmf";
            Eigen::MatrixXd A = readMatrixFromFile(AFilename);
            X.array() = X.array() / A.array();
        }

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_componentsVector(blockIndex), _pixels);
        Eigen::MatrixXd WBlock = Eigen::MatrixXd::Zero(_WClustered.rows(), 0);

        for (int j = 0; j < _nClusters; j++) {
            if (_componentPresences(blockIndex, j) == 1) {
                WBlock.conservativeResize(Eigen::NoChange, WBlock.cols() + 1);
                WBlock.col(WBlock.cols() - 1) = _WClustered.col(j);
            }
        }

        Eigen::MatrixXd GW = _G * WBlock;
        
        Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(GW);
        H = SVD.solve(X).cwiseAbs();

        Eigen::MatrixXd finalH = Eigen::MatrixXd::Zero(_nClusters, _pixels);
        
        int HIndex = 0;
        
        for (int j = 0; j < _nClusters; j++) {
            if (_componentPresences(blockIndex, j) == 1) {
                finalH.row(j) = H.row(HIndex);
                HIndex++;
            }
        }

        std::cout<<"Computed H block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";

        std::string HFilename = _outputDir + "/H/H_block_" + std::to_string(blockIndex) + ".onmf";
        writeMatrixToFile(finalH, HFilename);

        std::string QFilename = _outputDir + "/Q/Q_block_" + std::to_string(blockIndex) + ".onmf";
        Eigen::MatrixXd Q;
        
        if (!_model._quantificationElements.empty()) {
            Q = _model.generateQuantificationMatrix(_WClustered, finalH, _model._quantificationElements);
        }

        else {
            Q = _model.generateQuantificationMatrix(_WClustered, finalH);
        }

        writeMatrixToFile(Q, QFilename);
    }
}

void BlockWiseSmoothNMF::refineW( 
        const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
        SmoothNMFConstants::initialisation init, 
        int maxIter, 
        int randomSeed, 
        SmoothNMFConstants::algorithm algorithm,
        double tol, 
        double logShift, 
        double eps, 
        double lambdaL, 
        double mu, 
        double epsilonReg, 
        double dichotomyTol, 
        double sigmaL, 
        double gammaStepScalar,
        bool simplexW, 
        bool simplexH, 
        bool l2, 
        bool verbose, 
        bool safe,
        bool debug, 
        bool normalise, 
        bool noStopCriterion, 
        bool lineSearch
        )
    {
        std::cout<<"Refining the W matrix..."<<"\n\n";

        computeComponentPresences();

        for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
            std::string XFilename = _inputDir + "/X/X_block_" + std::to_string(blockIndex) + ".inmf";
            Eigen::MatrixXd X = readMatrixFromFile(XFilename);

            if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
                std::string AFilename = _outputDir + "/A/A_block_" + std::to_string(blockIndex) + ".onmf";
                _model._absorptionMatrixFilePath = AFilename;
            }

            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_componentsVector(blockIndex), _pixels);
            Eigen::MatrixXd fixedH = Eigen::MatrixXd::Constant(_componentsVector(blockIndex), _pixels, -1.0);
            Eigen::MatrixXd WBlock = Eigen::MatrixXd::Zero(_WClustered.rows(), 0);

            for (int j = 0; j < _nClusters; j++) {
                if (_componentPresences(blockIndex, j) == 1) {
                    WBlock.conservativeResize(Eigen::NoChange, WBlock.cols() + 1);
                    WBlock.col(WBlock.cols() - 1) = _WClustered.col(j);
                }
            }

            Eigen::MatrixXd fixedW = Eigen::MatrixXd::Constant(WBlock.rows(), WBlock.cols(), -1.0);
            
            SmoothNMF snmf = SmoothNMF(X, WBlock, H, fixedW, fixedH, gammaStepArray, 
                                    _componentsVector(blockIndex), init, maxIter, randomSeed, algorithm, 
                                    tol, logShift, eps, lambdaL, mu, epsilonReg, dichotomyTol, sigmaL, gammaStepScalar, 
                                    simplexW, simplexH, l2, verbose, safe, debug, normalise, noStopCriterion, lineSearch);

            snmf.setModel(_model);
            snmf.fitTransform();
            
            int WIndex = 0;
            
            for (int j = 0; j < _nClusters; j++) {
                if (_componentPresences(blockIndex, j) == 1) {
                    _WClustered.col(j) = snmf._W.col(WIndex);
                    WIndex++;
                }
            }

            std::cout<<"Refined W matrix on block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";
        }
}

void BlockWiseSmoothNMF::computeABlocks() {
    if (_WClustered.isZero()) {
        std::string WClusteredFilename = _outputDir + "/W/W_clustered.onmf";
        _WClustered = readMatrixFromFile(WClusteredFilename);
    }

    for (int blockIndex = 0; blockIndex < _blocks; blockIndex++) {
        std::string HFilename = _outputDir + "/H/H_block_" + std::to_string(blockIndex) + ".onmf";
        std::string TFilename = _inputDir + "/T/T_block_" + std::to_string(blockIndex) + ".inmf";
        _model._thicknessMapFilePath = TFilename;

        Eigen::MatrixXd HBlock = readMatrixFromFile(HFilename);
        Eigen::MatrixXd ABlock = _model.generateAbsorptionCorrectionMatrix(_WClustered, HBlock);
        
        std::cout<<"Computed A block : "<<blockIndex + 1<<" / "<<_blocks<<".\n\n";

        std::string AFilename = _outputDir + "/A/A_block_" + std::to_string(blockIndex) + ".onmf";
        writeMatrixToFile(ABlock, AFilename);
    }
}