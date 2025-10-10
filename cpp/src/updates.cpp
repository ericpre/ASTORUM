#include "../include/updates.h"

void initialiseNMF(const Eigen::Ref<const Eigen::MatrixXd>& X, int components, SmoothNMFConstants::initialisation init, double epsilon, int randSeed, bool debug, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H) {
    if (debug)
        std::cout<<"Entered initialiseNMF()."<<"\n";

    double averageX = X.mean();

    if (init == SmoothNMFConstants::initialisation::RANDOM) {
        if (debug)
            std::cout<<"Entered RANDOM case."<<"\n";

        std::mt19937 generator(randSeed);
        double stdMean = 0.0;
        double stdDev = 1.0;
        std::normal_distribution<double> normal(stdMean, stdDev);
        
        for (int i = 0; i < W.rows(); i++)
            for (int j = 0; j < W.cols(); j++)
                W(i, j) = std::abs(normal(generator)) * std::sqrt(averageX);
        
        for (int i = 0; i < H.rows(); i++)
            for (int j = 0; j < H.cols(); j++)
                H(i, j) = std::abs(normal(generator)) * std::sqrt(averageX);
        
        if (debug)
            std::cout<<"RANDOM case done."<<"\n";

        return;
    }

    if (init == SmoothNMFConstants::initialisation::NNDSVD || init == SmoothNMFConstants::initialisation::NNDSVDA || init == SmoothNMFConstants::initialisation::NNDSVDAR) {
        if (debug)
            std::cout<<"Entered NNDSVD* preliminary case."<<"\n";

        Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(X);

        Eigen::MatrixXd U = SVD.matrixU();
        Eigen::MatrixXd VT = SVD.matrixV().transpose();
        Eigen::VectorXd S = SVD.singularValues();

        if (debug) {
            std::cout<<"U.rows() = "<<U.rows()<<"\n";
            std::cout<<"U.cols() = "<<U.cols()<<"\n";
            std::cout<<"VT.rows() = "<<VT.rows()<<"\n";
            std::cout<<"VT.cols() = "<<VT.cols()<<"\n";
            std::cout<<"S.rows() = "<<S.rows()<<"\n";
            std::cout<<"S.cols() = "<<S.cols()<<"\n";
        }

        double sqrtS0 = std::sqrt(S(0));

        W.col(0).array() = U.col(0).array().abs() * sqrtS0;
        H.row(0).array() = VT.row(0).array().abs() * sqrtS0;

        for (int k = 1; k < components; k++) {
            Eigen::VectorXd x = U.col(k);
            Eigen::VectorXd y = VT.row(k);

            Eigen::VectorXd x_p = x.cwiseMax(0.0);
            Eigen::VectorXd y_p = y.cwiseMax(0.0);
            Eigen::VectorXd x_n = x.cwiseMin(0.0).cwiseAbs();
            Eigen::VectorXd y_n = y.cwiseMin(0.0).cwiseAbs();

            double x_p_nrm = x_p.norm();
            double y_p_nrm = y_p.norm();
            double x_n_nrm = x_n.norm();
            double y_n_nrm = y_n.norm();

            double m_p = x_p_nrm * y_p_nrm;
            double m_n = x_n_nrm * y_n_nrm;

            Eigen::VectorXd u;
            Eigen::VectorXd v;
            double sigma;

            if (m_p > m_n) {
                u = x_p / x_p_nrm;
                v = y_p / y_p_nrm;
                sigma = m_p;
            } 
            
            else {
                u = x_n / x_n_nrm;
                v = y_n / y_n_nrm;
                sigma = m_n;
            }

            double lambda = std::sqrt(S(k) * sigma);

            W.col(k) = lambda * u;
            H.row(k) = lambda * v;          
        }

        for (int i = 0; i < W.rows(); i++)
            for (int j = 0; j < W.cols(); j++)
                if (W(i ,j) < epsilon)
                    W(i, j) = 0.0;

        for (int i = 0; i < H.rows(); i++)
            for (int j = 0; j < H.cols(); j++)
                if (H(i, j) < epsilon)
                    H(i, j) = 0.0;
        
        if (debug)
            std::cout<<"NNDSVD* preliminary case done."<<"\n";
    }

    if (init == SmoothNMFConstants::initialisation::NNDSVD) {
        if (debug)
            std::cout<<"NNDSVD case done."<<"\n";
        return;
    }

    if (init == SmoothNMFConstants::initialisation::NNDSVDA) {
        for (int i = 0; i < W.rows(); i++)
            for (int j = 0; j < W.cols(); j++)
                if (W(i, j) == 0.0)
                    W(i, j) = averageX;
        
        for (int i = 0; i < H.rows(); i++)
            for (int j = 0; j < H.cols(); j++)
                if (H(i, j) == 0.0)
                    H(i, j) = averageX;

        if (debug)
            std::cout<<"NNDSVDA case done."<<"\n";
        
        return;
    }

    if (init == SmoothNMFConstants::initialisation::NNDSVDAR) {
        std::mt19937 generator(randSeed);
        double stdMean = 0.0;
        double stdDev = 1.0;
        std::normal_distribution<double> normal(stdMean, stdDev);

        for (int i = 0; i < W.rows(); i++)
            for (int j = 0; j < W.cols(); j++)
                if (W(i, j) == 0.0)
                    W(i, j) = std::abs(normal(generator) / 100);

        for (int i = 0; i < H.rows(); i++)
            for (int j = 0; j < H.cols(); j++)
                if (H(i, j) == 0.0)
                    H(i, j) = std::abs(normal(generator) / 100);

        if (debug)
            std::cout<<"NNDSVDAR case done."<<"\n";

        return;
    }
}

void initialiseAlgorithms(const Eigen::Ref<const Eigen::MatrixXd>& X, int components, SmoothNMFConstants::initialisation init, double epsilon, int randSeed, bool simplexW, bool simplexH, bool debug, double logShift, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<Eigen::MatrixXd> G, EDXSDataset& model) {
    if (debug)
        std::cout<<"Entered initialiseAlgorithms()."<<"\n";

    bool skipSecond = false;

    if (debug) {
        std::cout<<"G.rows() = "<<G.rows()<<"\n";
        std::cout<<"G.cols() = "<<G.cols()<<"\n";
    }

    if (G.isZero()) {
        if (debug)
            std::cout<<"Entered G.isZero() case."<<"\n";

        G = Eigen::VectorXd::Ones(X.rows()).asDiagonal();
        skipSecond = true;

        if (debug)
            std::cout<<"G.isZero() case done."<<"\n";
    }

    if (W.isZero()) {
        if (debug)
            std::cout<<"Entered W.isZero() case."<<"\n"; 

        Eigen::MatrixXd D(X.rows(), components);
        if (H.isZero()) {
            if (debug)
                std::cout<<"Entered H.isZero() case."<<"\n";

            initialiseNMF(X, components, init, epsilon, randSeed, debug, D, H);

            if (debug)
                std::cout<<"initialiseNMF() completed successfully."<<"\n";

            if (simplexH) {
                if (debug)
                    std::cout<<"Entered simplexH case."<<"\n";

                double factor = 1.0 / H.rows();

                for (int i = 0; i < H.rows(); i++)
                    for (int j = 0; j < H.cols(); j++)
                        if (std::isnan(H(i, j)))
                            H(i, j) = factor;
                
                Eigen::MatrixXd scaleMat = H.colwise().sum().replicate(H.rows(), 1);

                if (debug) {
                    std::cout<<"H.colwise.sum().rows() = "<<H.colwise().sum().rows()<<"\n";
                    std::cout<<"H.colwise.sum().cols() = "<<H.colwise().sum().cols()<<"\n";
                    std::cout<<"scaleMat.rows() = "<<scaleMat.rows()<<"\n";
                    std::cout<<"scaleMat.cols() = "<<scaleMat.cols()<<"\n";
                    std::cout<<"H.rows() = "<<H.rows()<<"\n";
                    std::cout<<"H.cols() = "<<H.cols()<<"\n";
                }
               
                H = H.cwiseQuotient(scaleMat).eval();

                double scaleMean = H.colwise().sum().mean();
                D *= scaleMean;

                if (debug)
                    std::cout<<"simplexH case done."<<"\n";
            }
            if (debug)
                std::cout<<"H.isZero() case done."<<"\n";
        }

        else {
            if (debug)
                std::cout<<"Entered non H.isZero() case."<<"\n";

            Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(H.transpose());
            Eigen::MatrixXd D = SVD.solve(X.transpose()).cwiseAbs().transpose();

            if (debug) {
                std::cout<<"D.rows() = "<<D.rows()<<"\n";
                std::cout<<"D.cols() = "<<D.cols()<<"\n";
                std::cout<<"non H.isZero() case done."<<"\n";
            }
        }

        if (skipSecond) {
            if (debug)
                std::cout<<"Entered skipSecond case."<<"\n";

            W = Eigen::MatrixXd::Zero(X.rows(), components);
            W = D;

            if (debug)
                std::cout<<"skipSecond case done."<<"\n";
        }

        else {
            if (debug)
                std::cout<<"Entered non skipSecond case."<<"\n";

            Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(G);
            W = SVD.solve(D).cwiseAbs();

            if (debug) {
                std::cout<<"W.rows() = "<<W.rows()<<"\n";
                std::cout<<"W.cols() = "<<W.cols()<<"\n";
            }

            if (simplexW) {
                if (debug)
                    std::cout<<"Entered simplexW case."<<"\n";

                double factor = 1.0 / W.rows();
                
                for (int i = 0; i < W.rows(); i++)
                    for (int j = 0; j < W.cols(); j++)
                        if (std::isnan(W(i, j)))
                            W(i, j) = factor;

                std::vector<int> simplexIndices = model.NMFSimplexIndices();

                Eigen::VectorXd scale = Eigen::VectorXd::Zero(W.cols());

                for (int j = 0; j < W.cols(); j++) {
                    double colScale = 0.0;

                    for (int idx : simplexIndices) {
                        colScale += W(idx, j);
                    }

                    for (int idx : simplexIndices) {
                        W(idx, j) /= colScale;
                    }
                }

                if (debug)
                    std::cout<<"simplexW case done."<<"\n";
            }

            if (debug)
                std::cout<<"non skipSecond case done."<<"\n";
        }

        if (debug)
            std::cout<<"W.isZero() case done."<<"\n";
    }

    else if (H.isZero()) {
        if (debug)
            std::cout<<"Entered non W.isZero() and H.isZero() case."<<"\n";

        Eigen::MatrixXd D = G * W;

        Eigen::BDCSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeThinV> SVD(D);
        H = SVD.solve(X).cwiseAbs();

        if (debug) {
            std::cout<<"H.rows() = "<<H.rows()<<"\n";
            std::cout<<"H.cols() = "<<H.cols()<<"\n";
        }

        if (simplexH) {
            if (debug)
                std::cout<<"Entered simplexH case."<<"\n";

            double factor = 1.0 / H.rows();

            for (int i = 0; i < H.rows(); i++)
                    for (int j = 0; j < H.cols(); j++)
                        if (std::isnan(H(i, j)))
                            H(i, j) = factor;

            if (debug) {
                std::cout<<"H.colwise().sum().rows() = "<<H.colwise().sum().rows()<<"\n";
                std::cout<<"H.colwise().sum().cols() = "<<H.colwise().sum().cols()<<"\n";
                std::cout<<"H.rows() = "<<H.rows()<<"\n";
                std::cout<<"H.cols() = "<<H.cols()<<"\n";
            }

            H = H.cwiseQuotient(H.colwise().sum().replicate(H.rows(), 1)).eval();

            if (debug)
                std::cout<<"simplexH case done."<<"\n";
        }

        if (debug)
            std::cout<<"non W.isZero() and H.isZero() case done."<<"\n";
    }

    W = W.cwiseMax(logShift).eval();
    H = H.cwiseMax(logShift).eval();

    if (debug)
        std::cout<<"initialiseAlgorithms() completed successfully."<<"\n";
}


void multiplicativeUpdateW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> H, bool simplexW, double logShift, bool safe, bool debug, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedW, bool useBregman, Eigen::Ref<Eigen::MatrixXd> W, EDXSDataset& model) {
    if (debug)
        std::cout<<"Entered multiplicativeUpdateW()."<<"\n";

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        double threshold = (-1) * logShift / 2;
        double sumH = 0.0;
        sumH = (H.array() < threshold).select(H, 0).sum();
        double sumW = 0.0;
        sumW = (W.array() < threshold).select(W, 0).sum();
        double sumG = 0.0;
        sumG = (G.array() < threshold).select(G, 0).sum();
        
        if (sumH || sumW || sumG)
            throw std::invalid_argument("Updates Error : Safe mode is not compatible with negative values in the input matrices.");

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    if (l2) {
        if (debug)
            std::cout<<"Entered l2 case."<<"\n";

        Eigen::MatrixXd GGWHH = ((G.transpose() * G) * W) * (H * H.transpose());
        Eigen::MatrixXd GTXHT = G.transpose() * (X * H.transpose());

        W.array() = (W.array() / GGWHH.array() * GTXHT.array()); 

        if (debug)
            std::cout<<"l2 case done."<<"\n";
    }

    else {
        if (debug)
            std::cout<<"Entered non l2 case."<<"\n";

        Eigen::MatrixXd GWH = (G * W) * H;
        Eigen::MatrixXd num;
        Eigen::MatrixXd denum;

        if (useBregman) {
            if (debug)
                std::cout<<"Entered useBregman case."<<"\n";

            if (G.isIdentity()) {
                if (debug)
                    std::cout<<"Entered G is identity case."<<"\n";
            }

            else {
                if (debug)
                    std::cout<<"Entered G is not identity case."<<"\n";

                double sigmaR = X.sum();
                Eigen::VectorXd Gsum = G.colwise().sum();
                Eigen::VectorXd Hsum = H.rowwise().sum();
                num = W.array() * sigmaR;
                Eigen::MatrixXd temp0 = X.cwiseQuotient(GWH);

                bool nanFlag = temp0.array().isNaN().any();
        
                if (nanFlag) {
                    if (debug)
                        std::cout<<"Entered nanFlag non-zero case."<<"\n";

                    GWH = GWH.cwiseMax(logShift).eval();
                    temp0 = X.cwiseQuotient(GWH);

                    if (debug)
                        std::cout<<"Treated nanFlag non-zero case."<<"\n";
                }

                Eigen::MatrixXd negativeG = (-1.0) * G.array();
                Eigen::MatrixXd gradG_part0 = negativeG.transpose() * temp0 * H.transpose();
                Eigen::MatrixXd gradG_part1 = Gsum * Hsum.transpose();
                Eigen::MatrixXd gradG = Eigen::MatrixXd(G.cols(), W.cols());
                gradG.array() = gradG_part0.array() / gradG_part1.array();
                denum.array() = gradG.array() * W.array() + sigmaR; 
            }
        }

        else {
            if (debug)
                std::cout<<"Entered non useBregman case."<<"\n";

            Eigen::MatrixXd temp0 = X.cwiseQuotient(GWH);
        
            bool nanFlag = temp0.array().isNaN().any();
        
            if (nanFlag) {
                if (debug)
                    std::cout<<"Entered nanFlag non-zero case."<<"\n";

                GWH = GWH.cwiseMax(logShift).eval();
                temp0 = X.cwiseQuotient(GWH);

                if (debug)
                    std::cout<<"Treated nanFlag non-zero case."<<"\n";
            }

            Eigen::MatrixXd temp1 = (G.transpose() * temp0) * H.transpose();
            num = W.cwiseProduct(temp1);
            Eigen::VectorXd Gsum = G.colwise().sum();
            Eigen::VectorXd Hsum = H.rowwise().sum();
            denum = Gsum * Hsum.transpose();

            if (debug) {
                std::cout<<"Gsum.rows() = "<<Gsum.rows()<<"\n";
                std::cout<<"Gsum.cols() = "<<Gsum.cols()<<"\n";
                std::cout<<"Hsum.rows() = "<<Hsum.rows()<<"\n";
                std::cout<<"Hsum.cols() = "<<Hsum.cols()<<"\n";
                std::cout<<"G.colwise().sum().rows() = "<<G.colwise().sum().rows()<<"\n";
                std::cout<<"G.colwise().sum().cols() = "<<G.colwise().sum().cols()<<"\n";
                std::cout<<"H.rowwise().sum().rows() = "<<H.rowwise().sum().rows()<<"\n";
                std::cout<<"H.rowwise().sum().cols() = "<<H.rowwise().sum().cols()<<"\n";
                std::cout<<"num.rows() = "<<num.rows()<<"\n";
                std::cout<<"num.cols() = "<<num.cols()<<"\n";
                std::cout<<"denum.rows() = "<<denum.rows()<<"\n";
                std::cout<<"denum.cols() = "<<denum.cols()<<"\n";
                std::cout<<"W.rows() = "<<W.rows()<<"\n";
                std::cout<<"W.cols() = "<<W.cols()<<"\n";
            }
        
            if (simplexW) {
                if (debug)
                    std::cout<<"Entered simplexW case."<<"\n";

                std::vector<int> simplexIndices = model.NMFSimplexIndices();

                if (debug)
                    std::cout<<"simplexIndices.size() = "<<simplexIndices.size()<<"\n";

                Eigen::MatrixXd numTemp = Eigen::MatrixXd::Zero(simplexIndices.size(), num.cols());
                Eigen::MatrixXd denumTemp = Eigen::MatrixXd::Zero(simplexIndices.size(), denum.cols());

                int tempIndex = 0;

                for (int idx : simplexIndices) {
                    if (debug)
                        std::cout<<"Simplex index = "<<idx<<"\n";

                    numTemp.row(tempIndex) = num.row(idx);
                    denumTemp.row(tempIndex) = denum.row(idx);
                    tempIndex++;
                }

                if (debug) {
                    std::cout<<"numTemp and denumTemp initialised."<<"\n";
                    std::cout<<"numTemp.rows() = "<<numTemp.rows()<<"\n";
                    std::cout<<"numTemp.cols() = "<<numTemp.cols()<<"\n";
                    std::cout<<"denumTemp.rows() = "<<denumTemp.rows()<<"\n";
                    std::cout<<"denumTemp.cols() = "<<denumTemp.cols()<<"\n";
                }

                Eigen::VectorXd nuTemp = Eigen::VectorXd::Zero(numTemp.cols());

                dichotomySimplex(numTemp, denumTemp, logShift, 1e-6, 100, nuTemp);

                if (debug) {
                    std::cout<<"numTemp.rows() = "<<numTemp.rows()<<"\n";
                    std::cout<<"numTemp.cols() = "<<numTemp.cols()<<"\n";
                    std::cout<<"denumTemp.rows() = "<<denumTemp.rows()<<"\n";
                    std::cout<<"denumTemp.cols() = "<<denumTemp.cols()<<"\n";
                    std::cout<<"nuTemp.rows() = "<<nuTemp.rows()<<"\n";
                    std::cout<<"nuTemp.cols() = "<<nuTemp.cols()<<"\n";
                }

                for (int idx : simplexIndices) {
                    denum.row(idx).array() += nuTemp.array().transpose();
                }

                if (debug)
                    std::cout<<"simplexW case done."<<"\n";
            }
        }
        
        W = num.cwiseQuotient(denum).cwiseMax(logShift);

        if ((fixedW.array() != -1.0).any())
            for (int i = 0; i < W.rows(); i++)
                for (int j = 0; j < W.cols(); j++)
                    if (fixedW(i, j) >= 0.0)
                        W(i, j) = fixedW(i, j);

        if (debug)
            std::cout<<"non l2 case done."<<"\n";
    }

    if (debug)
        std::cout<<"multiplicativeUpdateW() completed successfully."<<"\n";
}

void multiplicativeUpdateH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, bool simplexH, double mu, double epsilonReg, double lambdaL, double logShift, bool safe, bool debug, double dichotomyTol, double sigmaL, bool l2, bool useBregman, const Eigen::Ref<const Eigen::MatrixXd>& fixedH, Eigen::Ref<Eigen::MatrixXd> H) {
    if (debug)
        std::cout<<"Entered multiplicativeUpdateH()."<<"\n";

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        double threshold = (-1) * logShift / 2;
        double sumH = (H.array() < threshold).select(H, 0).sum();
        double sumW = (W.array() < threshold).select(W, 0).sum();
        double sumG = (G.array() < threshold).select(G, 0).sum();
        
        if (sumH || sumW || sumG)
            throw std::invalid_argument("Updates Error : Safe mode is not compatible with negative values in the input matrices.");

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    Eigen::MatrixXd D = G * W;
    Eigen::MatrixXd num(H.rows(), H.cols());
    Eigen::MatrixXd denum(H.rows(), H.cols());

    if (l2) {
        if (debug)
            std::cout<<"Entered l2 case."<<"\n";

        if (lambdaL != 0.0 || mu != 0.0)
            throw std::invalid_argument("Updates Error : lambdaL or mu cannot be used with l2 regularisation.");

        num = D.transpose() * X;
        denum = (D.transpose() * D) * H;

        if (debug) {
            std::cout<<"num.rows() = "<<num.rows()<<"\n";
            std::cout<<"num.cols() = "<<num.cols()<<"\n";
            std::cout<<"denum.rows() = "<<denum.rows()<<"\n";
            std::cout<<"denum.cols() = "<<denum.cols()<<"\n";
            std::cout<<"l2 case done."<<"\n";
        }

        if (debug)
            std::cout<<"l2 case done."<<"\n";
    }

    else {
        if (debug)
            std::cout<<"Entered non l2 case."<<"\n";

        Eigen::MatrixXd DH = D * H;

        if (useBregman) {
            if (debug)
                std::cout<<"Entered useBregman case."<<"\n";

            Eigen::VectorXd sigmaR = X.colwise().sum();

            if (debug) {
                std::cout<<"sigmaR.rows() = "<<sigmaR.rows()<<"\n";
                std::cout<<"sigmaR.cols() = "<<sigmaR.cols()<<"\n";
            }

            num = sigmaR.transpose().replicate(H.rows(), 1).array() / H.array();

            if (debug) {
                std::cout<<"num.rows() = "<<num.rows()<<"\n";
                std::cout<<"num.cols() = "<<num.cols()<<"\n";
            }

            Eigen::VectorXd Dsum = D.colwise().sum();
            Eigen::MatrixXd negativeD = (-1.0) * D.array();

            if (debug) {
                std::cout<<"negativeD.rows() = "<<negativeD.rows()<<"\n";
                std::cout<<"negativeD.cols() = "<<negativeD.cols()<<"\n";
            }

            Eigen::MatrixXd gradG_part0 = negativeD.transpose() * (X.cwiseQuotient(DH));
            Eigen::MatrixXd gradG_part1 = Dsum.replicate(1, H.cols());

            if (debug) {
                std::cout<<"gradG_part0.rows() = "<<gradG_part0.rows()<<"\n";
                std::cout<<"gradG_part0.cols() = "<<gradG_part0.cols()<<"\n";
                std::cout<<"gradG_part1.rows() = "<<gradG_part1.rows()<<"\n";
                std::cout<<"gradG_part1.cols() = "<<gradG_part1.cols()<<"\n";
            }

            Eigen::MatrixXd gradG = Eigen::MatrixXd(H.rows(), H.cols());
            gradG.array() = gradG_part0.array() + gradG_part1.array();
            denum.array() = gradG.array() + sigmaR.transpose().replicate(H.rows(), 1).array() / H.array();
        }

        else {
            if (debug)
                std::cout<<"Entered non useBregman case."<<"\n";
                
            num = D.transpose() * X.cwiseQuotient(DH);

            if (debug) {
                std::cout<<"num.rows() = "<<num.rows()<<"\n";
                std::cout<<"num.cols() = "<<num.cols()<<"\n";
            }
            
            bool nanFlag = num.array().isNaN().any();
            
            if (nanFlag) {
                if (debug)
                    std::cout<<"Entered nanFlag non-zero case."<<"\n";

                DH = DH.cwiseMax(logShift).eval();
                num = D.transpose() * X.cwiseQuotient(DH);

                if (debug)
                    std::cout<<"Treated nanFlag non-zero case."<<"\n";
            }

            Eigen::VectorXd Dsum = D.colwise().sum();
            denum = Dsum.replicate(1, H.cols());

            if (debug) {
                std::cout<<"D.colwise().sum().rows() = "<<D.colwise().sum().rows()<<"\n";
                std::cout<<"D.colwise().sum().cols() = "<<D.colwise().sum().cols()<<"\n";
                std::cout<<"Dsum.rows() = "<<Dsum.rows()<<"\n";
                std::cout<<"Dsum.cols() = "<<Dsum.cols()<<"\n";
                std::cout<<"denum.rows() = "<<denum.rows()<<"\n";
                std::cout<<"denum.cols() = "<<denum.cols()<<"\n";
                std::cout<<"H.rows() = "<<H.rows()<<"\n";
                std::cout<<"H.cols() = "<<H.cols()<<"\n";
                std::cout<<"non l2 case done."<<"\n";
            }
        }
    }

    if (mu != 0.0)
        denum.array() += mu / (H.array() + epsilonReg);
    
    if (lambdaL != 0.0) {
        if (debug)
            std::cout<<"Entered lambdaL != 0.0 case."<<"\n";

        Eigen::VectorXd HrowMax = H.rowwise().maxCoeff();
        Eigen::MatrixXd maxH = HrowMax.replicate(1, H.cols());

        if (debug) {
            std::cout<<"H.rowwise().maxCoeff().rows() = "<<H.rowwise().maxCoeff().rows()<<"\n";
            std::cout<<"H.rowwise().maxCoeff().cols() = "<<H.rowwise().maxCoeff().cols()<<"\n";
            std::cout<<"HrowMax.rows() = "<<HrowMax.rows()<<"\n";
            std::cout<<"HrowMax.cols() = "<<HrowMax.cols()<<"\n";
            std::cout<<"maxH.rows() = "<<maxH.rows()<<"\n";
            std::cout<<"maxH.cols() = "<<maxH.cols()<<"\n";
            std::cout<<"num.rows() = "<<num.rows()<<"\n";
            std::cout<<"num.cols() = "<<num.cols()<<"\n";
            std::cout<<"denum.rows() = "<<denum.rows()<<"\n";
            std::cout<<"denum.cols() = "<<denum.cols()<<"\n";
        }

        num.array() += lambdaL * sigmaL * maxH.array();
        denum.array() += lambdaL * sigmaL * maxH.array() + lambdaL * H.array();

        if (debug)
            std::cout<<"lambdaL != 0.0 case done."<<"\n";
    }

    num = num.cwiseProduct(H).eval();
    Eigen::MatrixXd nu(H.rows(), H.cols());

    if (simplexH) {
        if (debug)
            std::cout<<"Entered simplexH case."<<"\n";

        Eigen::VectorXd nuTemp(H.cols());
        dichotomySimplex(num, denum, logShift, dichotomyTol, 100, nuTemp);
        nu = nuTemp.transpose().replicate(H.rows(), 1);

        if (debug) {
            std::cout<<"nuTemp.rows() = "<<nuTemp.rows()<<"\n";
            std::cout<<"nuTemp.cols() = "<<nuTemp.cols()<<"\n";
            std::cout<<"nu.rows() = "<<nu.rows()<<"\n";
            std::cout<<"nu.cols() = "<<nu.cols()<<"\n";
            std::cout<<"simplexH case done."<<"\n";
        }
    }

    else {
        if (debug)
            std::cout<<"Entered non simplexH case."<<"\n";

        nu = Eigen::MatrixXd::Zero(H.rows(), H.cols());

        if (debug)
            std::cout<<"non simplexH case done."<<"\n";
    }

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        double sumDenum = 0.0;
        sumDenum = (denum.array() < 0.0).select(denum, 0).sum();
        double sumNum = 0.0;
        sumNum = (num.array() < 0.0).select(num, 0).sum();
        
        if (sumDenum != 0.0 || sumNum != 0.0)
            throw std::invalid_argument("Updates Error : Safe mode is not compatible with negative values in the input matrices.");

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    if (debug) {
        std::cout<<"Before updating H."<<"\n";
        std::cout<<"H.rows() = "<<H.rows()<<"\n";
        std::cout<<"H.cols() = "<<H.cols()<<"\n";
        std::cout<<"num.rows() = "<<num.rows()<<"\n";
        std::cout<<"num.cols() = "<<num.cols()<<"\n";
        std::cout<<"denum.rows() = "<<denum.rows()<<"\n";
        std::cout<<"denum.cols() = "<<denum.cols()<<"\n";
    }

    H.array() = num.array() / (denum.array() + nu.array());
    H = H.cwiseMax(logShift).eval();

    if ((fixedH.array() != -1.0).any())
        for (int i = 0; i < H.rows(); i++)
            for (int j = 0; j < H.cols(); j++)
                if (fixedH(i, j) >= 0.0)
                    H(i, j) = fixedH(i, j);

    if (debug)
        std::cout<<"multiplicativeUpdateH() completed successfully."<<"\n";
}

void gradW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double logShift, bool safe, bool debug, bool l2, Eigen::Ref<Eigen::MatrixXd> grad) {
    if (debug)
        std::cout<<"Entered gradW()."<<"\n";

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    if (l2) {
        if (debug)
            std::cout<<"Entered l2 case."<<"\n";

        grad = 2 * (G.transpose() * (((G * W) * H) - X)) * H.transpose();

        if (debug)
            std::cout<<"l2 case done."<<"\n";
    }

    else {
        if (debug)
            std::cout<<"Entered non l2 case."<<"\n";

        Eigen::MatrixXd DH = (G * W) * H;
        Eigen::MatrixXd temp0 = (-1) * X.cwiseQuotient(DH) * H.transpose();

        if (debug) {
            std::cout<<"H.rowwise().sum().rows() = "<<H.rowwise().sum().rows()<<"\n";
            std::cout<<"H.rowwise().sum().cols() = "<<H.rowwise().sum().cols()<<"\n";
            std::cout<<"temp0.rows() = "<<temp0.rows()<<"\n";
            std::cout<<"temp0.cols() = "<<temp0.cols()<<"\n";
        }

        Eigen::MatrixXd temp1;
        temp1.array() = temp0.array() + H.rowwise().sum().transpose().replicate(temp0.rows(), 1).array();
        grad = G.transpose() * temp1;

        if (debug)
            std::cout<<"non l2 case done."<<"\n";
    }

    if (debug)
        std::cout<<"gradW() completed successfully."<<"\n";
}

void gradH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, Eigen::Ref<Eigen::MatrixXd> H, double mu, double lambdaL, double epsilonReg, double logShift, bool safe, bool debug, bool l2, Eigen::Ref<Eigen::MatrixXd> grad) {
    if (debug)
        std::cout<<"Entered gradH()."<<"\n";

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    Eigen::MatrixXd D = G * W;

    if (l2) {
        if (debug)
            std::cout<<"Entered l2 case."<<"\n";

        grad = D.transpose() * ((D * H) - X);

        if (debug)
            std::cout<<"l2 case done."<<"\n";
    }

    else {
        if (debug)
            std::cout<<"Entered non l2 case."<<"\n";

        Eigen::MatrixXd DH = D * H;
        Eigen::MatrixXd temp0 = (-1) * D.transpose() * X.cwiseQuotient(DH);

        Eigen::VectorXd Dcolsum = D.colwise().sum();
        Eigen::MatrixXd Dsum(D.rows(), D.cols());
        Dsum = D.colwise().sum().transpose().replicate(1, H.cols());

        if (debug) {
            std::cout<<"D.colwise().sum().rows() = "<<D.colwise().sum().rows()<<"\n";
            std::cout<<"D.colwise().sum().cols() = "<<D.colwise().sum().cols()<<"\n";
            std::cout<<"Dcolsum.rows() = "<<Dcolsum.rows()<<"\n";
            std::cout<<"Dcolsum.cols() = "<<Dcolsum.cols()<<"\n";
            std::cout<<"Dsum.rows() = "<<Dsum.rows()<<"\n";
            std::cout<<"Dsum.cols() = "<<Dsum.cols()<<"\n";
            std::cout<<"D.rows() = "<<D.rows()<<"\n";
            std::cout<<"D.cols() = "<<D.cols()<<"\n";
        }

        grad.array() = temp0.array() + Dsum.array();

        if (debug)
            std::cout<<"non l2 case done."<<"\n";
    }

    if (mu != 0)
        grad.array() += mu / (H.array() + epsilonReg);

    if (lambdaL != 0)
        grad.array() += (lambdaL * H).array();

    if (debug)
        std::cout<<"gradH() completed successfully."<<"\n";
}

double estimateLipschitzBoundW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, int k, double logShift, bool debug) {
    if (debug)
        std::cout<<"Entered estimateLipschitzBoundW()."<<"\n";

    Eigen::MatrixXd Wlim = Eigen::MatrixXd::Constant(G.cols(), k, logShift);
    Eigen::MatrixXd Hlim = Eigen::MatrixXd::Constant(k, X.cols(), logShift);
    Eigen::MatrixXd DH = (G * Wlim) * Hlim;
    
    Eigen::VectorXd Hlimsumtemp = Hlim.colwise().sum();

    if (debug) {
        std::cout<<"Wlim.rows() = "<<Wlim.rows()<<"\n";
        std::cout<<"Wlim.cols() = "<<Wlim.cols()<<"\n";
        std::cout<<"Hlim.rows() = "<<Hlim.rows()<<"\n";
        std::cout<<"Hlim.cols() = "<<Hlim.cols()<<"\n";
        std::cout<<"Hlim.colwise().sum().rows() = "<<Hlim.colwise().sum().rows()<<"\n";
        std::cout<<"Hlim.colwise().sum().cols() = "<<Hlim.colwise().sum().cols()<<"\n";
        std::cout<<"Hlimsumtemp.rows() = "<<Hlimsumtemp.rows()<<"\n";
        std::cout<<"Hlimsumtemp.cols() = "<<Hlimsumtemp.cols()<<"\n";
    }

    Eigen::MatrixXd temp = (Hlim.colwise().sum().replicate(X.rows(), 1).cwiseProduct(X)).cwiseQuotient(DH.cwiseProduct(DH)) * Hlim.transpose();

    if (debug) {
        std::cout<<"temp.rows() = "<<temp.rows()<<"\n";
        std::cout<<"temp.cols() = "<<temp.cols()<<"\n";
        std::cout<<"estimateLipschitzBoundW() completed successfully."<<"\n";
    }

    return (temp.maxCoeff());
}

double estimateLipschitzBoundH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, int k, double logShift, double lambdaL, double mu, double epsilonReg, bool debug) {
    if (debug)
        std::cout<<"Entered estimateLipschitzBoundH()."<<"\n";
    
    Eigen::MatrixXd Wlim = Eigen::MatrixXd::Constant(G.cols(), k, logShift);
    Eigen::MatrixXd Hlim = Eigen::MatrixXd::Constant(k, X.cols(), logShift);
    Eigen::MatrixXd D = G * Wlim;
    Eigen::MatrixXd DH = D * Hlim;
    DH = DH.cwiseProduct(DH).eval();
    
    Eigen::VectorXd Drowsumtemp = D.rowwise().sum();

    if (debug) {
        std::cout<<"Wlim.rows() = "<<Wlim.rows()<<"\n";
        std::cout<<"Wlim.cols() = "<<Wlim.cols()<<"\n";
        std::cout<<"Hlim.rows() = "<<Hlim.rows()<<"\n";
        std::cout<<"Hlim.cols() = "<<Hlim.cols()<<"\n";
        std::cout<<"D.rowwise().sum().rows() = "<<D.rowwise().sum().rows()<<"\n";
        std::cout<<"D.rowwise().sum().cols() = "<<D.rowwise().sum().cols()<<"\n";
        std::cout<<"Dcolsumtemp.rows() = "<<Drowsumtemp.rows()<<"\n";
        std::cout<<"Dcolsumtemp.cols() = "<<Drowsumtemp.cols()<<"\n";
    }

    Eigen::MatrixXd temp = D.transpose() * (X.cwiseQuotient(DH).cwiseProduct(Drowsumtemp.replicate(1, X.cols())));

    if (debug) {
        std::cout<<"temp.rows() = "<<temp.rows()<<"\n";
        std::cout<<"temp.cols() = "<<temp.cols()<<"\n";
        std::cout<<"estimateLipschitzBoundH() completed successfully."<<"\n";
    }

    return (temp.maxCoeff() + 2 * lambdaL + mu * epsilonReg);
}

void projectedGradientStepW(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> H, double gamma, bool simplexW, double logShift, bool safe, bool debug, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedW, Eigen::Ref<Eigen::MatrixXd> W) {
    if (debug)
        std::cout<<"Entered projectedGradientStepW()."<<"\n";
    
    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    Eigen::MatrixXd grad(W.rows(), W.cols());
    gradW(X, G, W, H, logShift, safe, debug, l2, grad);

    if (debug) {
        std::cout<<"grad.rows() = "<<grad.rows()<<"\n";
        std::cout<<"grad.cols() = "<<grad.cols()<<"\n";
    }

    bool fixedWFlag = (fixedW.array() != -1.0).any();

    for (int i = 0; i < W.rows(); i++)
        for (int j = 0; j < W.cols(); j++) {
            W(i, j) -= 1/gamma * grad(i, j);

            if (fixedWFlag && fixedW(i, j) >= 0.0)
                W(i, j) = fixedW(i, j);
    }

    if (debug)
        std::cout<<"projectedGradientStepW() completed successfully."<<"\n";
}

void projectedGradientStepH(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& G, Eigen::Ref<Eigen::MatrixXd> W, double gamma, bool simplexH, double mu, double logShift, double epsilonReg, bool safe, bool debug, double dichotomyTol, double lambdaL, bool l2, const Eigen::Ref<const Eigen::MatrixXd>& fixedH, Eigen::Ref<Eigen::MatrixXd> H) {
    if (debug)
        std::cout<<"Entered projectedGradientStepH()."<<"\n";

    if (safe) {
        if (debug)
            std::cout<<"Entered safe mode."<<"\n";

        W = W.cwiseMax(logShift).eval();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"Treated safe mode."<<"\n";
    }

    Eigen::MatrixXd grad(H.rows(), H.cols());
    gradH(X, G, W, H, mu, lambdaL, epsilonReg, logShift, safe, debug, l2, grad);

    if (debug) {
        std::cout<<"grad.rows() = "<<grad.rows()<<"\n";
        std::cout<<"grad.cols() = "<<grad.cols()<<"\n";
    }
    
    H.array() -= 1/gamma * grad.array();

    Eigen::VectorXd nu(H.cols());

    if (simplexH) {
        if (debug)
            std::cout<<"Entered simplexH case."<<"\n";

        dichotomySimplexProjectedGradient(H, logShift, dichotomyTol, 100, nu);

        if (debug) {
            std::cout<<"nu.rows() = "<<nu.rows()<<"\n";
            std::cout<<"nu.cols() = "<<nu.cols()<<"\n";
            std::cout<<"H.rows() = "<<H.rows()<<"\n";
            std::cout<<"H.cols() = "<<H.cols()<<"\n";
        }

        H.array() += nu.transpose().replicate(H.rows(), 1).array();
        H = H.cwiseMax(logShift).eval();

        if (debug)
            std::cout<<"simplexH case done."<<"\n";
    }

    bool fixedHFlag = (fixedH.array() != -1.0).any();

    for (int i = 0; i < H.rows(); i++)
        for (int j = 0; j < H.cols(); j++) {
            if (fixedHFlag && fixedH(i, j) >= 0.0)
                H(i, j) = fixedH(i, j);
    }

    if (debug)
        std::cout<<"projectedGradientStepH() completed successfully."<<"\n";
}