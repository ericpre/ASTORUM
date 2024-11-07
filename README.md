# ASTORUM (**A**nalytical **ST**EM **O**ut-of-core **R**esource for **U**nified **M**ultimodal data)

## Description
ASTORUM is a robust and versatile data processing package for analytical scanning transmission electron microscopy (STEM) that specialises in the fusion of multiple STEM data modalities to enhance analytical accuracy and depth. Through its unique data integration framework, ASTORUM enables more precise insights by combining complementary datasets. For instance, by fusing energy-dispersive X-ray spectroscopy (EDXS) and electron energy-loss spectroscopy (EELS) low-loss data, ASTORUM achieves a more accurate absorption correction, fundamental for quantitative analysis. Alternatively, to refine absorption correction, the package can integrate EDXS with high-angle annular dark-field (HAADF) data. ASTORUM also supports out-of-core processing, allowing users to work with data sizes that exceed system memory limitations, a valuable capability for very large datasets.


The core codebase is written in C++ with the user interface and visualisation tools being written in Python.

The package is structured as follows:
- Core: C++ codebase containing the main computational algorithms and data processing pathways.
- Approximators: Non-negative matrix factorisation- (NMF) based hyperspectral unmixing tools specifically tailored for each type of supported STEM dataset.
- Utilities: Visualisation and helper tools.


## Features
- Block-wise out-of-core hyperspectral unmixing of EDXS datasets using an enhanced version of the SmoothNMF algorithm, originally available as part of the ESPM package, developed by _A. Teutrie et al._: https://github.com/adriente/espm.
- Usage of a refined X-ray absorption model for treatment of EDXS data by leveraging the relative thickness information from an EELS low-loss dataset.
- Refining the X-ray absorption model by incorporating HAADF information in the EDXS quantification data, inspired by the work of _J. Manassa et al._:  https://www.elementalmicroscopy.com/articles/EM000003/methods.

## Installation Prerequisites
<h3>Windows</h3>

- CMake: https://cmake.org/cmake/download.

- Visual Studio Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe.

## Installation
To install the package simply run the following command in the root directory of the repository:

    $ pip install -e .

## Getting Started

The repository includes all the required databases in addition to three experimental hyperspectral datasets (an EDXS dataset, an EELS low-loss dataset and a HAADF image) that can be downloaded.

Workflow example notebooks are included, showcasing some of the possible usages of ASTORUM.
Before running the notebooks, make sure you have PyQt6 installed:

    $ pip install pyqt6

Alternatively, use a different Qt-based matplotlib backend, such as Qt5 (_%matplotlib qt5_).
