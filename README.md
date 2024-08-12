# Block-Wise SmoothNMF

## Description
Block-wise hyperspectral data unmixing package based upon the the _SmoothNMF_ algorithm [https://github.com/adriente/espm].
The package is structured as follows:
- SmoothNMF : Direct translation of the _SmoothNMF_ algorithm in C++ with the main addition being an external absorption model for EDXS datasets that uses the EELS low-loss data simultaneously acquired with the EDXS data.
- BlockWiseSmoothNMF : A new block-wise approach, extending the _SmoothNMF_ algorithm in different directions, including out-of-core processing, enabling the identification of small minority phases in EDXS data and the external absorption model.
- PartitionTool : A visualisation tool used to identify the optimal partitioning scheme for block-wise approach.

Both decomposition algorithms are included in the *_BlockWiseSmoothNMFWrapper_* module and the partition tool is separated in *_PartitionTool_*.

## Installation
To install the package simply run the following command in the root directory of the repository:

    $ pip install -e .

## Getting Started

The repository comes loaded with all the required files in addition to two example hyperspectral datasets (an EDXS dataset and an EELS low-loss dataset).

Before running the workflow example notebook, make sure you have PyQt6 installed ($ pip install pyqt6) or use Qt5 backend for matplotlib instead (%matplotlib qt5).

