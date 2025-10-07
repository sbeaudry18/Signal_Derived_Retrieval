#### amf_calculator.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-29
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

############################

import numpy as np

def amf_calculator(no2_columns: np.ndarray, box_amfs: np.ndarray):
    if no2_columns.shape != box_amfs.shape:
        raise ValueError('no2_columns and box_amfs must have same shape. Input shapes were ' + str(no2_columns.shape) + ' and ' + str(box_amfs.shape) + 'respectively')
    amf = np.sum(no2_columns * box_amfs) / np.sum(no2_columns)
    return amf