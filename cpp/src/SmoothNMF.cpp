#include "../include/SmoothNMF.h"

void displayProgressBar(int currentIteration, int maxIterations, double loss, double elapsedSeconds) {
    int barWidth = 50;
    float progress = (float)currentIteration / maxIterations;
    int pos = barWidth * progress;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << "Iteration " << currentIteration << " / " << maxIterations 
              << " . Loss: " << loss 
              << " . Time elapsed: " << (int)(elapsedSeconds / 3600) << " h, "
              << (int)(fmod(elapsedSeconds, 3600) / 60) << " m and, " 
              << (int)(fmod(elapsedSeconds, 60)) << " s\r";
    std::cout.flush();
}

void rescaleDH(Eigen::Ref<Eigen::MatrixXd> D, Eigen::Ref<Eigen::MatrixXd> H) {
    Eigen::VectorXd O = Eigen::VectorXd::Ones(H.cols());
    Eigen::VectorXd X;

    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(H.transpose());
    X = SVD.solve(O);

    if ((X.array() <= 0.0).any()) {
        Eigen::NNLS<Eigen::MatrixXd> NNLS(H.transpose());
        X = NNLS.solve(O);
        X.array() = X.array().max(1E-10);
    }

    D *= X.cwiseInverse().asDiagonal();
    H = X.asDiagonal() * H;
}


NMFEstimator::NMFEstimator() {
    _channels = 0;
    _pixels = 0;
    _components = 0;
    _Gcols = 0;
    _init = SmoothNMFConstants::initialisation::RANDOM;
    _maxIter = 0;
    _randomSeed = 0;
    _nIter = 0;

    _tol = 0.0;
    _logShift = 0.0;
    _normFactor = 0.0;
    _constKL = 0.0;
    _eps = 0.0;
    _detailedLoss = 0.0;
    _reconstructionLoss = 0.0;

    _simplexW = false;
    _simplexH = false;
    _l2 = false;
    _verbose = true;
    _safe = false;
    _debug = false;
    _normalise = false;
    _noStopCriterion = false;

    _X = Eigen::MatrixXd::Zero(_channels, _pixels);
    _W = Eigen::MatrixXd::Zero(_Gcols, _components);
    _H = Eigen::MatrixXd::Zero(_components, _pixels);
    _G = Eigen::MatrixXd::Zero(_channels, _Gcols);
    _fixedW = Eigen::MatrixXd::Zero(_Gcols, _components);
    _fixedH = Eigen::MatrixXd::Zero(_components, _pixels);
    _L = Eigen::SparseMatrix<double>(_pixels, _pixels);
    _L.setIdentity();
};


NMFEstimator::NMFEstimator(
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        double logShift
        ) 
    {

    _channels = X.rows();
    _pixels = X.cols();
    _components = 2;
    _Gcols = 0;
    _init = SmoothNMFConstants::initialisation::RANDOM;
    _maxIter = 200;
    _randomSeed = 0;
    _nIter = 0;

    _tol = 1E-4;
    _logShift = logShift;
    _normFactor = 0.0;
    _constKL = 0.0;
    _eps = 1E-6;
    _detailedLoss = 0.0;
    _reconstructionLoss = 0.0;

    _simplexW = false;
    _simplexH = false;
    _l2 = false;
    _verbose = true;
    _safe = false;
    _debug = false;
    _normalise = false;
    _noStopCriterion = false;

    _X = X;
    _W = Eigen::MatrixXd::Zero(_Gcols, _components);
    _H = Eigen::MatrixXd::Zero(_components, _pixels);
    _G = Eigen::MatrixXd::Zero(_channels, _Gcols);
    _fixedW = Eigen::MatrixXd::Constant(_Gcols, _components, -1.0);
    _fixedH = Eigen::MatrixXd::Constant(_components, _pixels, -1.0);
    _L = Eigen::VectorXd::Ones(_pixels).asDiagonal();
};


NMFEstimator::NMFEstimator(
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        const Eigen::Ref<const Eigen::MatrixXd>& W, 
        const Eigen::Ref<const Eigen::MatrixXd>& H, 
        const Eigen::Ref<const Eigen::MatrixXd>& G, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
        int components, 
        SmoothNMFConstants::initialisation init, 
        int maxIter, 
        int randomSeed, 
        double tol, 
        double logShift, 
        double eps, 
        bool simplexW, 
        bool simplexH, 
        bool l2, 
        bool verbose, 
        bool safe, 
        bool debug, 
        bool normalise, 
        bool noStopCriterion
        ) 
    {   

    _channels = X.rows();
    _pixels = X.cols();
    _components = components;
    _Gcols = G.cols();
    _init = init;
    _maxIter = maxIter;
    _randomSeed = randomSeed;
    _nIter = 0;

    _tol = tol;
    _logShift = logShift;
    _normFactor = 0.0;
    _constKL = 0.0;
    _eps = eps;
    _detailedLoss = 0.0;
    _reconstructionLoss = 0.0;

    if (simplexW && simplexH)
        throw std::invalid_argument("Estimator Error : Cannot have both W and H constrained to simplex.");

    _simplexW = simplexW;
    _simplexH = simplexH;
    _l2 = l2;
    _verbose = verbose;
    _safe = safe;

    if (debug && maxIter > 1) {
        _maxIter = 1;
        std::cout<<"Debug mode requires maxIter = 1 but maxIter = "<<maxIter<<" was provided."<<"\n";
        std::cout<<"Setting maxIter = 1."<<"\n";
    }
        
    _debug = debug;
    _normalise = normalise;
    _noStopCriterion = noStopCriterion;

    _X = X;
    _W = W;
    _H = H;
    _G = G;
    _fixedW = fixedW;
    _fixedH = fixedH;
    _L = Eigen::VectorXd::Ones(_pixels).asDiagonal();
};


NMFEstimator::NMFEstimator(
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        const Eigen::Ref<const Eigen::MatrixXd>& W, 
        const Eigen::Ref<const Eigen::MatrixXd>& H, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
        int components, 
        SmoothNMFConstants::initialisation init, 
        int maxIter, 
        int randomSeed, 
        double tol, 
        double logShift, 
        double eps, 
        bool simplexW, 
        bool simplexH, 
        bool l2, 
        bool verbose, 
        bool safe, 
        bool debug, 
        bool normalise, 
        bool noStopCriterion
        ) 
    {   

    _channels = X.rows();
    _pixels = X.cols();
    _components = components;
    _Gcols = 0;
    _init = init;
    _maxIter = maxIter;
    _randomSeed = randomSeed;
    _nIter = 0;

    _tol = tol;
    _logShift = logShift;
    _normFactor = 0.0;
    _constKL = 0.0;
    _eps = eps;
    _detailedLoss = 0.0;
    _reconstructionLoss = 0.0;

    if (simplexW && simplexH)
        throw std::invalid_argument("Estimator Error : Cannot have both W and H constrained to simplex.");

    _simplexW = simplexW;
    _simplexH = simplexH;
    _l2 = l2;
    _verbose = verbose;
    _safe = safe;

    if (debug && maxIter > 1) {
        _maxIter = 1;
        std::cout<<"Debug mode requires maxIter = 1 but maxIter = "<<maxIter<<" was provided."<<"\n";
        std::cout<<"Setting maxIter = 1."<<"\n";
    }
        
    _debug = debug;
    _normalise = normalise;
    _noStopCriterion = noStopCriterion;

    _X = X;
    _W = W;
    _H = H;
    _G = Eigen::MatrixXd::Zero(0, 0);
    _fixedW = fixedW;
    _fixedH = fixedH;
    _L = Eigen::VectorXd::Ones(_pixels).asDiagonal();
};


NMFEstimator::~NMFEstimator() {
};


void NMFEstimator::initialiseModel(
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
}

void NMFEstimator::setModel(EDXSDataset model) {
    _model = model;
    _G = _model._G;
    _Gcols = _G.cols();
}

double NMFEstimator::normalisationFactor(const Eigen::Ref<const Eigen::MatrixXd>& X, double f) {   
    return f / (X.mean() *_channels);
}

void NMFEstimator::removeZeroLines(Eigen::Ref<Eigen::MatrixXd> X, double eps) {
    bool negativeVal = (X.array() < 0.0).any();

    if (negativeVal)
        throw std::runtime_error("Estimator Error : There are negative values in the original data matrix.");

    else {
        Eigen::VectorXd colSum = X.colwise().sum();
        Eigen::VectorXd rowSum = X.rowwise().sum();

        for (int i = 0; i < _channels; i++)
            for (int j = 0; j < _pixels; j++)
                if (rowSum(i) == 0 || colSum(j) == 0)
                    X(i, j) = eps;
    }
}

void NMFEstimator::iteration() {
    std::cerr<<"Iteration function not implemented."<<"\n";
}

double NMFEstimator::loss(Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, bool average) {
    double l = 0.0;

    Eigen::MatrixXd GW = _G * W;

    if (_l2)
        l = 0.5 * FrobeniusLoss(X, GW, _H, false);
    
    else {
        Eigen::MatrixXd XlogShiftMax = X.cwiseMax(_logShift);
        Eigen::MatrixXd Xlog;
        Xlog.array() = XlogShiftMax.array().log();
        Eigen::MatrixXd XXlog = Xlog.cwiseProduct(X);

        double constKL = XXlog.sum() - X.sum();
        l = KLDivLoss(X, GW, H, _logShift, false) + constKL;
    }

    if (average)
        l /= _channels * _pixels;

    return l;
}

void NMFEstimator::fitTransform() {
    if (_G.isZero()) {
        throw std::invalid_argument("Estimator Error : Please provide the G matrix in the constructor. Alternatively, initialise the model with the initialiseModel() function or provide a model with the setModel() function.");
    }

    if (_debug)
        std::cout<<"Entered fitTransform()."<<"\n";

    if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::EXTERNAL) {
        _model.applyAbsorptionCorrection(_X);
    }

    if (_safe)
        removeZeroLines(_X, _logShift);

    _normFactor = 0.0;

    if (_normalise) {
        _normFactor = normalisationFactor(_X, _components);
        _X = _X * _normFactor;
    }

    initialiseAlgorithms(_X, _components, _init, _eps, _randomSeed, _simplexW, _simplexH, _debug, _logShift, _W, _H, _G, _model);
    
    auto start = std::chrono::system_clock::now();
    double evalBefore = 1E20;
    double evalInit = loss(_X, _W, _H, true);
    _nIter = 0;

    while (true) {
        if (_debug)
            std::cout<<"Entered fitTransform() main loop."<<"\n";

        double tolMeanW;
        double tolMeanH;
        double evalAfter;
        double relW;
        double relH;
        
        Eigen::MatrixXd oldW = _W;
        Eigen::MatrixXd oldH = _H;

        iteration();

        evalAfter = loss(_X, _W, _H, true);
        _nIter++;

        Eigen::MatrixXd diffW = _W.array() - oldW.array();
        Eigen::MatrixXd diffH = _H.array() - oldH.array();
        diffW.array() = diffW.array().abs();
        diffH.array() = diffH.array().abs();

        tolMeanW = _tol * diffW.mean();
        tolMeanH = _tol * diffH.mean();

        Eigen::MatrixXd tempSumW = _W.array() + tolMeanW;
        Eigen::MatrixXd tempSumH = _H.array() + tolMeanH;

        Eigen::MatrixXd tW = diffW.cwiseQuotient(tempSumW);
        Eigen::MatrixXd tH = diffH.cwiseQuotient(tempSumH);

        relW = tW.maxCoeff();
        relH = tH.maxCoeff();

        if (_verbose && _nIter % 10 == 0) {
            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = now - start;
            double elapsedSeconds = duration.count();

            displayProgressBar(_nIter, _maxIter, evalAfter, elapsedSeconds);
        }

        if (_nIter >= _maxIter) {
            if (_verbose) {
                std::cout<<"\n";
            }

            std::cout<<"Maximum number of iterations reached."<<'\n';
            break;
        }

        if (!_noStopCriterion) {
            if (std::max(relW, relH) < _tol) {
                if (_verbose) {
                    std::cout<<"\n";
                }

                std::cout<<"Exit due to relative change "<<relH<<" and "<<relW<<" smaller than "<<_tol<<".\n";
                break;
            }

            else if (std::fabs((evalBefore - evalAfter) / evalInit) < _tol) {
                if (_verbose) {
                    std::cout<<"\n";
                }

                std::cout<<"Exit due to relative change "<<(evalBefore - evalAfter) / evalInit<<" smaller than "<<_tol<<".\n";
                break;
            }

            else if (evalAfter > evalBefore) {
                if (_verbose) {
                    std::cout<<"\n";
                }

                std::cout<<"Exit due to negative decrease "<<evalBefore - evalAfter<<" : "<<evalBefore<<" , "<<evalAfter<<".\n";
                break;
            }

            else if (std::isnan(evalAfter)) {
                if (_verbose) {
                    std::cout<<"\n";
                }

                std::cout<<"Exit due to NaN value."<<'\n';
                break;
            }
        }

        evalBefore = evalAfter;
    }

    if (_verbose) {
        std::cout<<"\n";
    }

    if (!_simplexW && !_simplexH) {
        rescaleDH(_W, _H);
        if (_debug)
            std::cout<<"Rescaled W and H in non-simplex case."<<"\n";
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    double elapsedSeconds = duration.count();

    std::cout<<"Stopped after "<<_nIter<<" iterations in "<<(int)(elapsedSeconds / 3600)<<" hours, "<<(int)(fmod(elapsedSeconds, 3600) / 60)<<" minutes and, "<< (int)(fmod(elapsedSeconds, 60))<<" seconds."<<"\n";

    _reconstructionLoss = loss(_X, _W, _H, true);

    if (_normalise)
        _W /= _normFactor;
    
    if (_debug) {
        std::cout<<"Reconstruction loss = "<<_reconstructionLoss<<"\n";
        std::cout<<"Finished fitTransform()."<<"\n";
    }
}


SmoothNMF::SmoothNMF() : NMFEstimator() {
    _gammaStepArray = Eigen::VectorXd::Zero(2);

    _algorithm = SmoothNMFConstants::algorithm::LOG_SURROGATE;

    _lambdaL = 0.0;
    _mu = 0.0;
    _epsilonReg = 0.0;
    _dichotomyTol = 0.0;
    _sigmaL = 0.0;
    _gammaStepScalar = 0.0;
    
    _simplexW = false;
    _simplexH = false;
    _lineSearch = false;
};


SmoothNMF::SmoothNMF(
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        double logShift
        ) : 
        NMFEstimator(
            X, 
            logShift
            ) 
    {

    _gammaStepArray = Eigen::VectorXd::Zero(2);

    _algorithm = SmoothNMFConstants::algorithm::LOG_SURROGATE;

    _lambdaL = 1.0;
    _mu = 0.0;
    _epsilonReg = 1.0;
    _dichotomyTol = 1E-3;
    _sigmaL = 8.0;
    _gammaStepScalar = 0.0;
    
    _simplexW = true;
    _simplexH = false;
    _lineSearch = false;
};


SmoothNMF::SmoothNMF(                
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        const Eigen::Ref<const Eigen::MatrixXd>& W, 
        const Eigen::Ref<const Eigen::MatrixXd>& H, 
        const Eigen::Ref<const Eigen::MatrixXd>& G, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
        const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
        int components,  
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
        ) : 
        NMFEstimator(
            X, 
            W, 
            H, 
            G, 
            fixedW, 
            fixedH, 
            components, 
            init, 
            maxIter, 
            randomSeed, 
            tol, 
            logShift, 
            eps, 
            simplexW, 
            simplexH, 
            l2, 
            verbose, 
            safe, 
            debug, 
            normalise, 
            noStopCriterion
            ) 
        {
        
        _gammaStepArray = gammaStepArray;

        _algorithm = algorithm;

        _lambdaL = lambdaL;
        _mu = mu;
        _epsilonReg = epsilonReg;
        _dichotomyTol = dichotomyTol;
        _sigmaL = sigmaL;
        _gammaStepScalar = gammaStepScalar;

        _lineSearch = lineSearch;
};


SmoothNMF::SmoothNMF(                
        const Eigen::Ref<const Eigen::MatrixXd>& X, 
        const Eigen::Ref<const Eigen::MatrixXd>& W, 
        const Eigen::Ref<const Eigen::MatrixXd>& H,
        const Eigen::Ref<const Eigen::MatrixXd>& fixedW, 
        const Eigen::Ref<const Eigen::MatrixXd>& fixedH, 
        const Eigen::Ref<const Eigen::VectorXd>& gammaStepArray,
        int components,  
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
        ) : 
        NMFEstimator(
            X, 
            W, 
            H, 
            fixedW, 
            fixedH, 
            components, 
            init, 
            maxIter, 
            randomSeed, 
            tol, 
            logShift, 
            eps, 
            simplexW, 
            simplexH, 
            l2, 
            verbose, 
            safe, 
            debug, 
            normalise, 
            noStopCriterion
            ) 
        {
        
        _gammaStepArray = gammaStepArray;

        _algorithm = algorithm;

        _lambdaL = lambdaL;
        _mu = mu;
        _epsilonReg = epsilonReg;
        _dichotomyTol = dichotomyTol;
        _sigmaL = sigmaL;
        _gammaStepScalar = gammaStepScalar;

        _lineSearch = lineSearch;
};


SmoothNMF::~SmoothNMF() {
};


double SmoothNMF::lossSmoothNMF(Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, bool average) {
    double lkl = loss(_X, W, H, average);
    double reg = logReg(H, _mu, _epsilonReg, average);

    if (average)
        reg /= _channels * _pixels;

    Eigen::MatrixXd HT = H.transpose();
    double l2 = 0.5 * _lambdaL * traceXTLX(_L, HT, false);

    if (average)
        l2 /= _channels * _pixels;
    
    return (lkl + reg + l2);
}

void SmoothNMF::iteration() {
    if (_debug)
        std::cout<<"Entered iteration()."<<"\n";
        
    if (_nIter == 0) {
        if (_gammaStepScalar == 0.0 && _gammaStepArray.isZero()) {
            if (_algorithm == SmoothNMFConstants::algorithm::LOG_SURROGATE || _algorithm == SmoothNMFConstants::algorithm::L2_SURROGATE || _algorithm == SmoothNMFConstants::algorithm::BMD)
                _gammaStepScalar = _sigmaL;

            else if (_algorithm == SmoothNMFConstants::algorithm::PROJECTED_GRADIENT) {
                double gammaW = estimateLipschitzBoundW(_X, _G, _components, _logShift, _debug);
                double gammaH = estimateLipschitzBoundH(_X, _G, _components, _logShift, _lambdaL, _mu, _epsilonReg, _debug);

                if (_debug) {
                    std::cout<<"Projected gradient case; iteration 0."<<"\n";
                    std::cout<<"gammaW = "<<gammaW<<"\n";
                    std::cout<<"gammaH = "<<gammaH<<"\n";
                }

                _gammaStepArray(0) = gammaH;
                _gammaStepArray(1) = gammaW;
            }
        }
    }

    if (_debug)
        std::cout<<"Updating H..."<<"\n";

    Eigen::MatrixXd Hold;

    if (_lineSearch)
        Hold = _H;

    if (_algorithm == SmoothNMFConstants::algorithm::L2_SURROGATE) {
        // To be implemented
    }

    else if (_algorithm == SmoothNMFConstants::algorithm::LOG_SURROGATE) {
        if (_debug)
            std::cout<<"Entered LOG_SURROGATE case."<<"\n";
        multiplicativeUpdateH(_X, _G, _W, _simplexH, _mu, _epsilonReg, _lambdaL, _logShift, _safe, _debug, _dichotomyTol, _sigmaL, _l2, false, _L, _fixedH, _H);
    }

    else if (_algorithm == SmoothNMFConstants::algorithm::PROJECTED_GRADIENT) {
        if (_debug)
            std::cout<<"Entered PROJECTED_GRADIENT case."<<"\n";
        projectedGradientStepH(_X, _G, _W, _gammaStepArray(0), _simplexH, _mu, _logShift, _epsilonReg, _safe, _debug, _dichotomyTol, _lambdaL, _L, _l2, _fixedH, _H);
    }

    else if (_algorithm == SmoothNMFConstants::algorithm::BMD) {
        if (_debug)
            std::cout<<"Entered BMD case."<<"\n";
        multiplicativeUpdateH(_X, _G, _W, _simplexH, _mu, _epsilonReg, _lambdaL, _logShift, _safe, _debug, _dichotomyTol, _sigmaL, _l2, true, _L, _fixedH, _H);
    }

    else
        throw std::invalid_argument("Estimator Error : Invalid algorithm.");

    if (_lineSearch) {
        if (_debug)
            std::cout<<"Entered lineSearch case."<<"\n";
        if (_algorithm == SmoothNMFConstants::algorithm::L2_SURROGATE || _algorithm == SmoothNMFConstants::algorithm::LOG_SURROGATE || _algorithm == SmoothNMFConstants::algorithm::BMD) {
            if (_debug)
                std::cout<<"Entered LOG_SURROGATE case."<<"\n";

            double d = diffSurrogate(Hold, _H, _L, _sigmaL, _lambdaL, _algorithm);

            if (_debug)
                std::cout<<"d = "<<d<<"\n";
            
            if (d > 0)
                _gammaStepScalar /= 1.05;
            
            else
                _gammaStepScalar *= 1.5;
        }

        else {
            if (_debug)
                std::cout<<"Entered PROJECTED_GRADIENT case."<<"\n";

            Eigen::MatrixXd grad;
            gradH(_X, _G, _W, Hold, _mu, _lambdaL, _L, _epsilonReg, _logShift, _safe, _debug, _l2, grad);

            double lossOld = lossSmoothNMF(_W, Hold, false);

            if (_debug)
                std::cout<<"lossOld = "<<lossOld<<"\n";

            double lossNew = lossSmoothNMF(_W, _H, false);

            if (_debug)
                std::cout<<"lossNew = "<<lossNew<<"\n";

            double qSurrogate = quadraticSurrogate(_H, Hold, grad, lossOld, _gammaStepArray[0]);

            if (_debug)
                std::cout<<"qSurrogate = "<<qSurrogate<<"\n";

            double d = qSurrogate - lossNew;

            if (d > 0)
                _gammaStepArray[0] /= 1.05;
            
            else
                _gammaStepArray[0] *= 1.5;
        }
    }

    if (_debug)
        std::cout<<"H updated sucessfully."<<"\n";

    if (_debug)
        std::cout<<"Updating W..."<<"\n";
    Eigen::MatrixXd Wold;

    if (_algorithm == SmoothNMFConstants::algorithm::L2_SURROGATE || _algorithm == SmoothNMFConstants::algorithm::LOG_SURROGATE) {
        if (_debug)
            std::cout<<"Entered L2_SURROGATE or LOG_SURROGATE case."<<"\n";
        multiplicativeUpdateW(_X, _G, _H, _simplexW, _logShift, _safe, _debug, _l2, _fixedW, false, _W, _model);
    }

    else if (_algorithm == SmoothNMFConstants::algorithm::BMD) {
        if (_debug)
            std::cout<<"Entered BMD case."<<"\n";
        multiplicativeUpdateW(_X, _G, _H, _simplexW, _logShift, _safe, _debug, _l2, _fixedW, true, _W, _model);
    }

    else if (_algorithm == SmoothNMFConstants::algorithm::PROJECTED_GRADIENT) {
        if (_debug)
            std::cout<<"Entered PROJECTED_GRADIENT case."<<"\n";

        if (_lineSearch)
            Wold = _W;

        projectedGradientStepW(_X, _G, _H, _gammaStepArray[1], _simplexW, _logShift, _safe, _debug, _l2, _fixedW, _W);

        if (_lineSearch) {
            if (_debug)
                std::cout<<"Entered lineSearch case."<<"\n";

            Eigen::MatrixXd grad;
            gradW(_X, _G, Wold, _H, _logShift, _safe, _debug, _l2, grad);

            double lossOld = lossSmoothNMF(Wold, _H, false);

            if (_debug)
                std::cout<<"lossOld = "<<lossOld<<"\n";

            double lossNew = lossSmoothNMF(_W, _H, false);

            if (_debug)
                std::cout<<"lossNew = "<<lossNew<<"\n";

            double qSurrogate = quadraticSurrogate(_W, Wold, grad, lossOld, _gammaStepArray[1]);

            if (_debug)
                std::cout<<"qSurrogate = "<<qSurrogate<<"\n";

            double d = qSurrogate - lossNew;

            if (d > 0)
                _gammaStepArray[1] /= 1.05;
            
            else
                _gammaStepArray[1] *= 1.5;
        }
    }

    if (_model._absorptionModelType == EDXSModelConstants::absorptionModelType::INTERNAL) {
        _model.updateGBremsstrahlung(_W);
    }

    if (_debug) {
        std::cout<<"W updated sucessfully."<<"\n";
        std::cout<<"Finished iteration()."<<"\n";
    }
}

void SmoothNMF::fitTransform() {
    NMFEstimator::fitTransform();
}