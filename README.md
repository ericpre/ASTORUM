# Block-Wise SmoothNMF

## Description
Block-wise hyperspectral data unmixing package based upon the _SmoothNMF_ algorithm [https://github.com/adriente/espm].

The package is structured as follows:
- SmoothNMF: Direct translation of the _SmoothNMF_ algorithm in C++ with the main addition being an external absorption model for EDXS datasets that uses the EELS low-loss data simultaneously acquired with the EDXS data.
- BlockWiseSmoothNMF: A new block-wise approach, extending the _SmoothNMF_ algorithm in different directions, including out-of-core processing, enabling the identification of small minority phases in EDXS data and leveraging the external absorption model.
- PartitionTool: A visualisation tool used to identify the optimal partitioning scheme and to estimate the number of relevant components per block for usage with the block-wise approach.

Both decomposition algorithms are part in the main *_BlockWiseSmoothNMFWrapper_* module while the partitioning tool is implemented in the *_PartitionTool_* submodule.

## Installation Prerequisites
<h3>Windows Prerequisites</h3>
On Windows platforms, prior installation of CMake and Visual Studio Build Tools is required.

- CMake: https://cmake.org/cmake/download.

- Visual Studio Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe.

## Installation
To install the package simply run the following command in the root directory of the repository:

    $ pip install -e .

## Getting Started

The repository includes all the required database files in addition to two experimental hyperspectral datasets (an EDXS dataset and an EELS low-loss dataset) that can be downloaded by running the first section from the *examples/workflow.ipynb* Jupyter notebook.

The workflow example notebook showcases some of the possible usages of _BlockWiseSmoothNMF_.
Before running the notebook, make sure you have PyQt6 installed in your environment:

    $ pip install pyqt6

Alternatively, use a different Qt-based matplotlib backend, such as Qt5 (_%matplotlib qt5_).
