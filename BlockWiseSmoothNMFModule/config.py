import os
import sys
import hyperspy.api as hs
import dask.array as da
import exspy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATABASE_DIR = ROOT_DIR / Path("databases")

LIBRARY_DIR = ROOT_DIR / Path("cpp/out")

DETECTOR_EFFICIENCY = DATABASE_DIR / Path("interpolated_SDD_efficiency.txt")
PERIODIC_TABLE_INFO = DATABASE_DIR / Path("periodic_table_symbols.json")
PERIODIC_TABLE_NUMBERS = DATABASE_DIR / Path("periodic_table_number.json")
SIEGBAHN_TO_IUPAC = DATABASE_DIR / Path("siegbahn_to_iupac.json")
MASS_ABSORPTION_COEFFICIENTS = DATABASE_DIR / Path("interpolated_mass_absorption_coefficients.json")
XRAY_200KeV = DATABASE_DIR / Path("200keV_xrays_transformed.json")
XRAY_300KeV = DATABASE_DIR / Path("300keV_xrays_transformed.json")

sys.path.append(str(LIBRARY_DIR))