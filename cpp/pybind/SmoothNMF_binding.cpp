#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include "../include/BlockWiseSmoothNMF.h"
#include "../include/EELSDataset.h"

PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);

namespace py = pybind11;


PYBIND11_MODULE(BlockWiseSmoothNMFlib, m) {
    py::bind_vector<std::vector<int>>(m, "IntVector");
    py::bind_vector<std::vector<double>>(m, "DoubleVector");
    py::bind_vector<std::vector<std::string>>(m, "StringVector");
    py::bind_vector<std::vector<std::vector<int>>>(m, "IntVector2D");

    py::enum_<EDXSModelConstants::meanType>(m, "EDXSModelConstants_meanType")
        .value("HARMONIC", EDXSModelConstants::meanType::HARMONIC)
        .value("WEIGHTED", EDXSModelConstants::meanType::WEIGHTED);

    py::enum_<EDXSModelConstants::problemType>(m, "EDXSModelConstants_problemType")
        .value("IDENTITY", EDXSModelConstants::problemType::IDENTITY)
        .value("NO_BREMSSTRAHLUNG", EDXSModelConstants::problemType::NO_BREMSSTRAHLUNG)
        .value("BREMSSTRAHLUNG", EDXSModelConstants::problemType::BREMSSTRAHLUNG);

    py::enum_<EDXSModelConstants::absorptionModelType>(m, "EDXSModelConstants_absorptionModelType")
        .value("INTERNAL", EDXSModelConstants::absorptionModelType::INTERNAL)
        .value("EXTERNAL", EDXSModelConstants::absorptionModelType::EXTERNAL)
        .value("NONE", EDXSModelConstants::absorptionModelType::NONE);


    py::enum_<SmoothNMFConstants::algorithm>(m, "SmoothNMFConstants_algorithm")
        .value("LOG_SURROGATE", SmoothNMFConstants::algorithm::LOG_SURROGATE)
        .value("L2_SURROGATE", SmoothNMFConstants::algorithm::L2_SURROGATE)
        .value("PROJECTED_GRADIENT", SmoothNMFConstants::algorithm::PROJECTED_GRADIENT)
        .value("BMD", SmoothNMFConstants::algorithm::BMD);

    py::enum_<SmoothNMFConstants::initialisation>(m, "SmoothNMFConstants_initialisation")
        .value("RANDOM", SmoothNMFConstants::initialisation::RANDOM)
        .value("NNDSVD", SmoothNMFConstants::initialisation::NNDSVD)
        .value("NNDSVDA", SmoothNMFConstants::initialisation::NNDSVDA)
        .value("NNDSVDAR", SmoothNMFConstants::initialisation::NNDSVDAR);


    py::enum_<BlockWiseSmoothNMFConstants::distanceMetric>(m, "BlockWiseSmoothNMFConstants_distanceMetric")
        .value("EUCLIDEAN", BlockWiseSmoothNMFConstants::distanceMetric::EUCLIDEAN)
        .value("MANHATTAN", BlockWiseSmoothNMFConstants::distanceMetric::MANHATTAN)
        .value("MAHALANOBIS", BlockWiseSmoothNMFConstants::distanceMetric::MAHALANOBIS)
        .value("MINKOWSKI", BlockWiseSmoothNMFConstants::distanceMetric::MINKOWSKI)
        .value("COSINE", BlockWiseSmoothNMFConstants::distanceMetric::COSINE)
        .value("CHEBYSHEV", BlockWiseSmoothNMFConstants::distanceMetric::CHEBYSHEV)
        .value("HAMMING", BlockWiseSmoothNMFConstants::distanceMetric::HAMMING)
        .value("JACCARD", BlockWiseSmoothNMFConstants::distanceMetric::JACCARD);

    m.doc() = "A C++ version of the ESPM library for the analysis of EDXS data.";
    m.def("countGcolumns", &countGColumns, "Compute the number of columns for the G matrix based on elements and split lines elements.", py::arg("elements"), py::arg("splitLinesElements"));
    m.def("printConcentrationReport", &printConcentrationReport, "Print the concentration report for each phase.", py::arg("G"), py::arg("W"), py::arg("H"), py::arg("modelElements"), py::arg("selectedElements"), py::arg("fitError"),
        py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    m.def("energyToArrayIndex", &energyToArrayIndex, "Convert energy to array index.", py::arg("energy"), py::arg("energyAxisScale"), py::arg("energyAxisOffset"));
    m.def("estimateThicknessAtPixel", &estimateThicknessAtPixel, "Estimate the thickness at a pixel.", py::arg("EELSLowLossSpectrum"), py::arg("energyAxis"), py::arg("energyAxisScale"), py::arg("energyAxisOffset"), py::arg("zeroLossPeakThreshold"), py::arg("density"), py::arg("electronEnergy"), py::arg("alpha"), py::arg("beta"));


    py::class_<EDXSDataset>(m, "EDXSDataset")
        .def(py::init<
            int ,
            EDXSModelConstants::problemType ,
            EDXSModelConstants::absorptionModelType ,
            double ,
            double ,
            double ,
            double ,
            double ,
            double ,
            double ,
            double ,
            double ,
            double ,
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::vector<std::string> ,
            std::vector<std::string> ,
            std::vector<double> ,
            std::vector<std::string> ,
            std::vector<std::string> , 
            py::EigenDRef<Eigen::VectorXd> 
            >())

        .def("generateGMatrix", &EDXSDataset::generateGMatrix, "Generate the G matrix.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("NMFSimplexIndices", &EDXSDataset::NMFSimplexIndices, "Get the indices of the G matrix rows correspnding to the elements used for the simplex constraint.")
        .def("NMFSimplexElements", &EDXSDataset::NMFSimplexElements, "Get the elements used for the simplex constraint.")
        .def("selectedElementsIndices", &EDXSDataset::selectedElementsIndices, "Get the indices of the G matrix rows corresponding to the selected elements.", py::arg("selectedElements"))
        .def("updateGBremsstrahlung", &EDXSDataset::updateGBremsstrahlung, "Update the Bremsstrahlung part of the G matrix based on the internal absorption model.", py::arg("W"))
        .def("readThicknessMap", &EDXSDataset::readThicknessMap, "Read the EELS thickness map from a file.")
        .def("generateQuantificationMatrix", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd>&, const Eigen::Ref<const Eigen::MatrixXd>&>(&EDXSDataset::generateQuantificationMatrix), "Generate a quantification matrix based on the W and H matrices obtained from a previous NMF decomposition.", py::arg("W"), py::arg("H"))
        .def("generateQuantificationMatrix", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd>&, const Eigen::Ref<const Eigen::MatrixXd>&, const std::vector<std::string>&>(&EDXSDataset::generateQuantificationMatrix), "Generate a quantification matrix based on the W and H matrices obtained from a previous NMF decomposition with specified elements used for quantification.", py::arg("W"), py::arg("H"), py::arg("selectedElements"))
        .def("generateAbsorptionCorrectionMatrix", &EDXSDataset::generateAbsorptionCorrectionMatrix, "Generate an absorption correction matrix based on the W and H matrices obtained from a previous NMF decomposition. The matrix factors are applied to experimental data to correct for the absorption effects.", py::arg("W"), py::arg("H"))
        .def("computeDensityMap", &EDXSDataset::computeDensityMap, "Compute the density map based on the W and H matrices obtained from a previous NMF decomposition.", py::arg("W"), py::arg("H"))
        .def("applyAbsorptionCorrection", &EDXSDataset::applyAbsorptionCorrection, "Apply the absorption correction to experimental data.", py::arg("X"))
        
        .def_readwrite("Gcols", &EDXSDataset::_Gcols)
        .def_readwrite("beamEnergy", &EDXSDataset::_beamEnergy)
        .def_readwrite("problemType", &EDXSDataset::_problemType)
        .def_readwrite("absorptionModelType", &EDXSDataset::_absorptionModelType)
        .def_readwrite("azimuthAngle", &EDXSDataset::_azimuthAngle)
        .def_readwrite("elevationAngle", &EDXSDataset::_elevationAngle)
        .def_readwrite("tiltStage", &EDXSDataset::_tiltStage)
        .def_readwrite("thickness", &EDXSDataset::_thickness)
        .def_readwrite("density", &EDXSDataset::_density)
        .def_readwrite("widthSlope", &EDXSDataset::_widthSlope)
        .def_readwrite("widthIntercept", &EDXSDataset::_widthIntercept)
        .def_readwrite("energyAxisSize", &EDXSDataset::_energyAxisSize)
        .def_readwrite("energyAxisScale", &EDXSDataset::_energyAxisScale)
        .def_readwrite("energyAxisOffset", &EDXSDataset::_energyAxisOffset)
        .def_readwrite("detectorEfficiency", &EDXSDataset::_detectorEfficiency)
        .def_readwrite("xrayDB", &EDXSDataset::_xrayDB)
        .def_readwrite("massAbsorptionCoefficientsFilePath", &EDXSDataset::_massAbsorptionCoefficientsFilePath)
        .def_readwrite("decompositionResultsFilePath", &EDXSDataset::_decompositionResultsFilePath)
        .def_readwrite("absorptionMatrixFilePath", &EDXSDataset::_absorptionMatrixFilePath)
        .def_readwrite("thicknessMapFilePath", &EDXSDataset::_thicknessMapFilePath)
        .def_readwrite("periodicTableInfoFilePath", &EDXSDataset::_periodicTableInfoFilePath)
        .def_readwrite("xrayDBFile", &EDXSDataset::_xrayDBFile)
        .def_readwrite("massAbsorptionCoefficientsDBFile", &EDXSDataset::_massAbsorptionCoefficientsDBFile)
        .def_readwrite("periodicTableInfoDBFile", &EDXSDataset::_periodicTableInfoDBFile)
        .def_readwrite("elements", &EDXSDataset::_elements)
        .def_readwrite("splitLinesElements", &EDXSDataset::_splitLinesElements)
        .def_readwrite("G", &EDXSDataset::_G)
        .def_readwrite("modelElements", &EDXSDataset::_modelElements)
        .def_readwrite("quantificationElements", &EDXSDataset::_quantificationElements)
        .def_readwrite("absorptionElements", &EDXSDataset::_absorptionElements)
        .def_readwrite("absorptionElementsConcentrations", &EDXSDataset::_absorptionElementsConcentrations);


    py::class_<EELSDataset>(m, "EELSDataset")
        .def(py::init<
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::VectorXd> ,
            double ,
            double ,
            double ,
            py::EigenDRef<Eigen::VectorXd> ,
            double ,
            double ,
            double >())

        .def("computeThicknessMap", &EELSDataset::computeThicknessMap, "Compute the thickness map based on the EELS low-loss data.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def_readwrite("X", &EELSDataset::_X)
        .def_readwrite("energyAxis", &EELSDataset::_energyAxis)
        .def_readwrite("channels", &EELSDataset::_channels)
        .def_readwrite("pixels", &EELSDataset::_pixels)
        .def_readwrite("energyAxisScale", &EELSDataset::_energyAxisScale)
        .def_readwrite("energyAxisOffset", &EELSDataset::_energyAxisOffset)
        .def_readwrite("electronEnergy", &EELSDataset::_electronEnergy)
        .def_readwrite("densityMap", &EELSDataset::_densityMap)
        .def_readwrite("alpha", &EELSDataset::_alpha)
        .def_readwrite("beta", &EELSDataset::_beta)
        .def_readwrite("zeroLossPeakThreshold", &EELSDataset::_zeroLossPeakThreshold)
        .def_readwrite("T", &EELSDataset::_T);


    py::class_<SmoothNMF>(m, "SmoothNMF")
        .def(py::init<
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            int , 
            SmoothNMFConstants::initialisation , 
            int , 
            int , 
            SmoothNMFConstants::algorithm , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double ,
            bool , 
            bool , 
            bool , 
            bool , 
            bool ,
            bool , 
            bool , 
            bool ,  
            bool >())

        .def(py::init<
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            py::EigenDRef<Eigen::MatrixXd> ,
            int , 
            SmoothNMFConstants::initialisation , 
            int , 
            int , 
            SmoothNMFConstants::algorithm , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double , 
            double ,
            bool , 
            bool , 
            bool , 
            bool , 
            bool ,
            bool , 
            bool , 
            bool ,  
            bool >())

        .def("fitTransform", &SmoothNMF::fitTransform, "Run the NMF algorithm to fit the model to the data.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("initialiseModel", &SmoothNMF::initialiseModel, "Initialise the underlying EDXSDataset model.", 
                py::arg("beamEnergy"), py::arg("problemType"), py::arg("absorptionModelType"), py::arg("azimuthAngle"), py::arg("elevationAngle"), 
                py::arg("tiltStage"), py::arg("thickness"), py::arg("density"), py::arg("widthSlope"), py::arg("widthIntercept"), py::arg("energyAxisSize"), 
                py::arg("energyAxisScale"), py::arg("energyAxisOffset"), py::arg("detectorEfficiency"), py::arg("xrayDB"), py::arg("massAbsorptionCoefficientsFilePath"),
                py::arg("decompositionResultsFilePath"), py::arg("absorptionMatrixFilePath"), py::arg("thicknessMapFilePath"),
                py::arg("periodicTableInfoFilePath"), py::arg("elements"), py::arg("splitLinesElements"), py::arg("energyThresholds"),
                py::arg("quantificationElements"), py::arg("absorptionElements"), py::arg("absorptionElementsConcentrations"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def_readwrite("X", &SmoothNMF::_X)
        .def_readwrite("W", &SmoothNMF::_W)
        .def_readwrite("H", &SmoothNMF::_H)
        .def_readwrite("G", &SmoothNMF::_G)
        .def_readwrite("fixedW", &SmoothNMF::_fixedW)
        .def_readwrite("fixedH", &SmoothNMF::_fixedH)
        .def_readwrite("L", &SmoothNMF::_L)
        .def_readwrite("channels", &SmoothNMF::_channels)
        .def_readwrite("pixels", &SmoothNMF::_pixels)
        .def_readwrite("components", &SmoothNMF::_components)
        .def_readwrite("Gcols", &SmoothNMF::_Gcols)
        .def_readwrite("init", &SmoothNMF::_init)
        .def_readwrite("maxIter", &SmoothNMF::_maxIter)
        .def_readwrite("randomSeed", &SmoothNMF::_randomSeed)
        .def_readwrite("maxIter", &SmoothNMF::_maxIter)
        .def_readwrite("tol", &SmoothNMF::_tol)
        .def_readwrite("logShift", &SmoothNMF::_logShift)
        .def_readwrite("normFactor", &SmoothNMF::_normFactor)
        .def_readwrite("constKL", &SmoothNMF::_constKL)
        .def_readwrite("eps", &SmoothNMF::_eps)
        .def_readwrite("detailedLoss", &SmoothNMF::_detailedLoss)
        .def_readwrite("reconstructionLoss", &SmoothNMF::_reconstructionLoss)
        .def_readwrite("simplexW", &SmoothNMF::_simplexW)
        .def_readwrite("simplexH", &SmoothNMF::_simplexH)
        .def_readwrite("l2", &SmoothNMF::_l2)
        .def_readwrite("verbose", &SmoothNMF::_verbose)
        .def_readwrite("safe", &SmoothNMF::_safe)
        .def_readwrite("debug", &SmoothNMF::_debug)
        .def_readwrite("normalise", &SmoothNMF::_normalise)
        .def_readwrite("noStopCriterion", &SmoothNMF::_noStopCriterion)
        .def_readwrite("gammaStepArray", &SmoothNMF::_gammaStepArray)
        .def_readwrite("algorithm", &SmoothNMF::_algorithm)
        .def_readwrite("lambdaL", &SmoothNMF::_lambdaL)
        .def_readwrite("mu", &SmoothNMF::_mu)
        .def_readwrite("epsilonReg", &SmoothNMF::_epsilonReg)
        .def_readwrite("dichotomyTol", &SmoothNMF::_dichotomyTol)
        .def_readwrite("sigmaL", &SmoothNMF::_sigmaL)
        .def_readwrite("gammaStepScalar", &SmoothNMF::_gammaStepScalar)
        .def_readwrite("lineSearch", &SmoothNMF::_lineSearch)
        .def_readwrite("model", &SmoothNMF::_model);


    py::class_<BlockWiseSmoothNMF>(m, "BlockWiseSmoothNMF")
        .def(py::init<
            const std::string& ,
            const std::string& ,
            int ,
            int ,
            int ,
            int ,
            int ,
            py::EigenDRef<Eigen::VectorXi>
            >())

        .def("initialiseModel", &BlockWiseSmoothNMF::initialiseModel, "Initialise the underlying EDXSDataset model.", 
                py::arg("beamEnergy"), py::arg("problemType"), py::arg("absorptionModelType"), py::arg("azimuthAngle"), py::arg("elevationAngle"), 
                py::arg("tiltStage"), py::arg("thickness"), py::arg("density"), py::arg("widthSlope"), py::arg("widthIntercept"), py::arg("energyAxisSize"), 
                py::arg("energyAxisScale"), py::arg("energyAxisOffset"), py::arg("detectorEfficiency"), py::arg("xrayDB"), py::arg("massAbsorptionCoefficientsFilePath"),
                py::arg("decompositionResultsFilePath"), py::arg("absorptionMatrixFilePath"), py::arg("thicknessMapFilePath"),
                py::arg("periodicTableInfoFilePath"), py::arg("elements"), py::arg("splitLinesElements"), py::arg("energyThresholds"), py::arg("quantificationElements"), py::arg("absorptionElements"), py::arg("absorptionElementsConcentrations"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("setModel", &BlockWiseSmoothNMF::setModel, "Set the underlying EDXSDataset model.", py::arg("model"))
        .def("computeWBlocks", &BlockWiseSmoothNMF::computeWBlocks, "Compute the W blocks.", 
                py::arg("gammaStepArray"), py::arg("init"), py::arg("maxIter"), py::arg("randomSeed"), py::arg("algorithm"), 
                py::arg("tol"), py::arg("logShift"), py::arg("eps"), py::arg("lambdaL"), py::arg("mu"), py::arg("epsilonReg"), 
                py::arg("dichotomyTol"), py::arg("sigmaL"), py::arg("gammaStepScalar"), py::arg("simplexW"), py::arg("simplexH"), 
                py::arg("l2"), py::arg("verbose"), py::arg("safe"), py::arg("debug"), py::arg("normalise"), py::arg("noStopCriterion"), 
                py::arg("lineSearch"), py::arg("separationOrder"), py::arg("precomputedW"), py::arg("writeWblocks"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("spectralClustering", &BlockWiseSmoothNMF::spectralClustering, "Perform spectral clustering on the W blocks.", 
                py::arg("clusteringSigma"), py::arg("metric"), py::arg("p"), py::arg("clusteringIter"), py::arg("clusteringTolerance"))
        .def("computeClusteredW", &BlockWiseSmoothNMF::computeClusteredW, "Compute the clustered W matrix.")
        .def("VCA", &BlockWiseSmoothNMF::VCA, "Perform vertex component analysis (VCA) on the W blocks.", py::arg("verbose"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeAbundanceVCA", &BlockWiseSmoothNMF::computeAbundanceVCA, "Compute the abundance matrix of the VCA endmembers.")
        .def("printClusterLabels", &BlockWiseSmoothNMF::printClusterLabels, "Print the cluster labels.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("getMonolithicX", &BlockWiseSmoothNMF::getMonolithicX, "Get the monolithic X matrix.")
        .def("computeHBlocks", &BlockWiseSmoothNMF::computeHBlocks, "Compute the H blocks using SmoothNMF with fixed W.", 
                py::arg("gammaStepArray"), py::arg("init"), py::arg("maxIter"), py::arg("randomSeed"), py::arg("algorithm"), 
                py::arg("tol"), py::arg("logShift"), py::arg("eps"), py::arg("lambdaL"), py::arg("mu"), py::arg("epsilonReg"), 
                py::arg("dichotomyTol"), py::arg("sigmaL"), py::arg("gammaStepScalar"), py::arg("simplexW"), py::arg("simplexH"), 
                py::arg("l2"), py::arg("verbose"), py::arg("safe"), py::arg("debug"), py::arg("normalise"), py::arg("noStopCriterion"), 
                py::arg("lineSearch"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeHMatrix", &BlockWiseSmoothNMF::computeHMatrix, "Compute the monolithic H matrix using SmoothNMF with fixed W.", 
                py::arg("gammaStepArray"), py::arg("init"), py::arg("maxIter"), py::arg("randomSeed"), py::arg("algorithm"), 
                py::arg("tol"), py::arg("logShift"), py::arg("eps"), py::arg("lambdaL"), py::arg("mu"), py::arg("epsilonReg"), 
                py::arg("dichotomyTol"), py::arg("sigmaL"), py::arg("gammaStepScalar"), py::arg("simplexW"), py::arg("simplexH"), 
                py::arg("l2"), py::arg("verbose"), py::arg("safe"), py::arg("debug"), py::arg("normalise"), py::arg("noStopCriterion"), 
                py::arg("lineSearch"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeHBlocksSVD", &BlockWiseSmoothNMF::computeHBlocksSVD, "Compute the H blocks using SVD.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeHMatrixSVD", &BlockWiseSmoothNMF::computeHMatrixSVD, "Compute the monolithic H matrix using SVD.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("refineW", &BlockWiseSmoothNMF::refineW, "Refine the clustered W matrix.", 
                py::arg("gammaStepArray"), py::arg("init"), py::arg("maxIter"), py::arg("randomSeed"), py::arg("algorithm"), 
                py::arg("tol"), py::arg("logShift"), py::arg("eps"), py::arg("lambdaL"), py::arg("mu"), py::arg("epsilonReg"), 
                py::arg("dichotomyTol"), py::arg("sigmaL"), py::arg("gammaStepScalar"), py::arg("simplexW"), py::arg("simplexH"), 
                py::arg("l2"), py::arg("verbose"), py::arg("safe"), py::arg("debug"), py::arg("normalise"), py::arg("noStopCriterion"), 
                py::arg("lineSearch"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("refineWVCA", &BlockWiseSmoothNMF::refineWVCA, "Refine the W matrix obtained using VCA.", 
                py::arg("gammaStepArray"), py::arg("init"), py::arg("maxIter"), py::arg("randomSeed"), py::arg("algorithm"), 
                py::arg("tol"), py::arg("logShift"), py::arg("eps"), py::arg("lambdaL"), py::arg("mu"), py::arg("epsilonReg"), 
                py::arg("dichotomyTol"), py::arg("sigmaL"), py::arg("gammaStepScalar"), py::arg("simplexW"), py::arg("simplexH"), 
                py::arg("l2"), py::arg("verbose"), py::arg("safe"), py::arg("debug"), py::arg("normalise"), py::arg("noStopCriterion"), 
                py::arg("lineSearch"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeABlocks", &BlockWiseSmoothNMF::computeABlocks, "Compute the absorption correction matrix blocks.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("computeAMatrix", &BlockWiseSmoothNMF::computeAMatrix, "Compute the monolithic absorption correction matrix.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        
        .def_readwrite("inputDir", &BlockWiseSmoothNMF::_inputDir)
        .def_readwrite("outputDir", &BlockWiseSmoothNMF::_outputDir)
        .def_readwrite("blocksHorizontal", &BlockWiseSmoothNMF::_blocksHorizontal)
        .def_readwrite("blocksVertical", &BlockWiseSmoothNMF::_blocksVertical)
        .def_readwrite("blocks", &BlockWiseSmoothNMF::_blocks)
        .def_readwrite("pixels", &BlockWiseSmoothNMF::_pixels)
        .def_readwrite("blockWidth", &BlockWiseSmoothNMF::_blockWidth)
        .def_readwrite("blockHeight", &BlockWiseSmoothNMF::_blockHeight)
        .def_readwrite("Gcols", &BlockWiseSmoothNMF::_Gcols)
        .def_readwrite("nPoints", &BlockWiseSmoothNMF::_nPoints)
        .def_readwrite("nDims", &BlockWiseSmoothNMF::_nDims)
        .def_readwrite("nClusters", &BlockWiseSmoothNMF::_nClusters)
        .def_readwrite("componentsVector", &BlockWiseSmoothNMF::_componentsVector)
        .def_readwrite("WElements", &BlockWiseSmoothNMF::_WElements)
        .def_readwrite("WBremsstrahlung", &BlockWiseSmoothNMF::_WBremsstrahlung)
        .def_readwrite("WFull", &BlockWiseSmoothNMF::_WFull)
        .def_readwrite("WClustered", &BlockWiseSmoothNMF::_WClustered)
        .def_readwrite("clusterLabels", &BlockWiseSmoothNMF::_clusterLabels)
        .def_readwrite("componentPresences", &BlockWiseSmoothNMF::_componentPresences)
        .def_readwrite("model", &BlockWiseSmoothNMF::_model)
        .def_readwrite("G", &BlockWiseSmoothNMF::_G);
}