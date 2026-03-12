#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace EDXSModelConstants
{
    enum class meanType : int {
        HARMONIC = 0x60,
        WEIGHTED = 0x61
    };

    enum class problemType : int {
        IDENTITY = 0x70,
        NO_BREMSSTRAHLUNG = 0x71,
        BREMSSTRAHLUNG = 0x72
    };

    enum class absorptionModelType : int {
        INTERNAL = 0x80,
        EXTERNAL = 0x81,
        NONE = 0x82
    };
}

namespace EELSModelConstants
{
    enum class electronMeanFreePathComputation : int {
        DENSITY_OF_MIXTURE = 0x90,
        MEAN_ATOMIC_NUMBER = 0x91
    };
}

namespace SmoothNMFConstants 
{
    enum class algorithm : int {
        LOG_SURROGATE = 0x100,
        L2_SURROGATE = 0x101,
        PROJECTED_GRADIENT = 0x102,
        BMD = 0x103
    };

    enum class initialisation : int {
        RANDOM = 0x110,
        NNDSVD = 0x111,
        NNDSVDA = 0x112,
        NNDSVDAR = 0x113
    };
}

namespace BlockWiseSmoothNMFConstants
{
    enum class distanceMetric : int {
        EUCLIDEAN = 0x200, 
        MANHATTAN = 0x201, 
        MAHALANOBIS = 0x202, 
        MINKOWSKI = 0x203, 
        COSINE = 0x204, 
        CHEBYSHEV = 0x205, 
        HAMMING = 0x206, 
        JACCARD = 0x207
    };

    enum class fusionType : int {
        EELS = 0x210,
        HAADF = 0x211
    };
}

#endif