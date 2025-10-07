#### cloud_lyr_amf_adjustment.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-29
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

#####################################

import numpy as np

def cloud_lyr_amf_adjustment(upper_interface_pressure, lower_interface_pressure, p_cloud):
    '''
    Finds correction factor for box AMF from the independent pixel approximation by accounting for the portion of the cloudy part of pixel below the cloud pressure

    upper_interface_pressure: pressure at upper interface of cloud layer (hPa)
    lower_interface_pressure: pressure at lower interace of cloud layer (hPa)
    p_cloud: cloud pressure (hPa)
    '''

    # The cloud layer is a three-part combination 
    diff_p_upper_minus_lower = np.abs(upper_interface_pressure - lower_interface_pressure)
    diff_p_upper_minus_cloud = np.abs(upper_interface_pressure - p_cloud)
    f_above_cloud = diff_p_upper_minus_cloud / diff_p_upper_minus_lower
    f_below_cloud = 1 - f_above_cloud
    #                                    # Clear         # Cloudy     # Above cloud non-zero # Below cloud zero
    #b_amf_calculated = ((1 - w) * b_amf_clear) + (w * ((f_above_cloud * b_amf_cloudy) + (f_below_cloud * 0)))

    return f_above_cloud